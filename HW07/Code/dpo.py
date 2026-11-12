"""
Direct Preference Optimization (Rafailov et al., 2023; lecture slide 11) with LoRA, for both the text
LLM (Part C basic) and the VLM (Part C advanced).

  loss = -log sigmoid( beta * [ (log pi(y_w|x) - log pi_ref(y_w|x)) - (log pi(y_l|x) - log pi_ref(y_l|x)) ] )

- Policy = frozen FP16 base model + trainable LoRA adapters (PEFT). Reference = the same model with the
  adapters disabled, so only one copy of the 4 B weights is in GPU memory (fits the 11 GB 2080 Ti).
- Reference log-probs are computed once before training (as TRL's precompute_ref_log_probs).
- log pi(y|x) = sum of the log-probabilities of the response tokens only (prompt tokens are masked).
  `logits_to_keep` makes the LM head run on the response positions only, which saves ~1 GB of logits.
- FP16 autocast + GradScaler on T4 / 2080 Ti (no native bf16), FP32 on CPU (smoke tests).
This is the same objective as TRL's DPOTrainer (loss_type="sigmoid"); a small custom loop is used so
that one implementation serves both the LLM and the VLM.
"""

import contextlib
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .common import DEVICE, save_json, load_json

# attention + MLP projections of the language model only (vision encoder stays frozen)
LORA_TARGETS = r'^(?!.*(visual|vision)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$'


def _ignore_incompatible_torchao():
    """Colab preinstalls torchao 0.10; recent peft raises ImportError for torchao < 0.16 while it builds LoRA
    layers, although LoRA on a plain FP16 model never uses torchao. Treat an incompatible torchao as absent."""
    import importlib
    try:
        importlib.import_module('peft.import_utils').is_torchao_available()
        return
    except (ImportError, AttributeError):
        pass
    for name in ('peft.import_utils', 'peft.tuners.lora.torchao', 'peft.utils.quantization_utils'):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(mod, 'is_torchao_available'):
            mod.is_torchao_available = lambda: False


def add_lora(model, r=16, alpha=32, dropout=0.05):
    _ignore_incompatible_torchao()
    from peft import LoraConfig, get_peft_model
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=LORA_TARGETS, bias='none')
    pm = get_peft_model(model, cfg)
    n_train = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in pm.parameters())
    print(f'[dpo] LoRA r={r}: {n_train / 1e6:.2f} M trainable / {n_all / 1e9:.2f} B total parameters')
    return pm


def load_lora(model, adapter_dir):
    _ignore_incompatible_torchao()
    from peft import PeftModel
    model.config.use_cache = True
    return PeftModel.from_pretrained(model, adapter_dir).eval()


def _base_dtype(model):
    return next(p.dtype for p in model.parameters() if p.is_floating_point())   # embedding = base weights' dtype


def _autocast(model):
    dt = _base_dtype(model)
    if DEVICE.type != 'cuda' or dt == torch.float32:
        return contextlib.nullcontext(), False
    return torch.autocast('cuda', dtype=dt), dt == torch.float16


def sequence_logp(model, inputs, n_resp):
    """Sum of log p(response tokens | prompt); the response is the last n_resp tokens of input_ids."""
    out = model(**inputs, logits_to_keep=n_resp + 1, use_cache=False)
    logits = out.logits[:, -(n_resp + 1):-1].float()
    labels = inputs['input_ids'][:, -n_resp:]
    return torch.gather(logits.log_softmax(-1), 2, labels.unsqueeze(-1)).squeeze(-1).sum(-1)


def _pair_logps(model, encode, pair, ctx):
    """(log pi(chosen), log pi(rejected), number of chosen response tokens)."""
    lp, n_c = [], None
    for which in ('chosen', 'rejected'):
        inputs, n = encode(pair, which)
        n_c = n if which == 'chosen' else n_c
        with ctx:
            lp.append(sequence_logp(model, inputs, n))
    return lp[0], lp[1], n_c


@torch.no_grad()
def reference_logps(model, pairs, encode):
    ctx, _ = _autocast(model)
    model.eval()
    out = []
    with model.disable_adapter():
        for p in pairs:
            c, r, _ = _pair_logps(model, encode, p, ctx)
            out.append((c.item(), r.item()))
    return out


@torch.no_grad()
def evaluate_pairs(model, pairs, refs, encode, beta):
    """DPO loss / reward accuracy / margin, plus how far log pi(chosen) and log pi(rejected) moved from the
    reference (a negative chosen log-ratio = likelihood displacement: the preferred answer also lost probability)."""
    ctx, _ = _autocast(model)
    model.eval()
    rows = []
    for p, (rc, rr) in zip(pairs, refs):
        c, r, n = _pair_logps(model, encode, p, ctx)
        dc, dr = c.item() - rc, r.item() - rr
        z = beta * (dc - dr)
        rows.append((float(-F.logsigmoid(torch.tensor(z))), z > 0, dc - dr, dc, dr, -c.item() / n))
    a = np.array(rows, dtype=float)
    return {'loss': a[:, 0].mean(), 'reward_acc': a[:, 1].mean(), 'margin': a[:, 2].mean(),
            'chosen_logratio': a[:, 3].mean(), 'rejected_logratio': a[:, 4].mean(), 'chosen_nll_per_token': a[:, 5].mean()}


def train_dpo(model, train_pairs, val_pairs, encode, out_dir, beta=0.1, lr=5e-5, epochs=2, grad_accum=8,
              sft_alpha=0.0, max_grad_norm=1.0, warmup_ratio=0.1, seed=42, log_every=10, save_every=25):
    """model: PEFT model (LoRA added). encode(pair, 'chosen'|'rejected') -> (inputs on device, n_response_tokens).
    sft_alpha > 0 adds alpha x (per-token NLL of the chosen answer) to the DPO loss (RPO, Pang et al. 2024;
    TRL rpo_alpha), which keeps the likelihood of the preferred answer from collapsing together with the rejected one.
    Keeps the adapter of the epoch with the lowest validation DPO loss; writes out_dir/history.json.
    Resumable: reference log-probs are cached in out_dir/refs.pt and training continues from resume.pt."""
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from . import trainer
    os.makedirs(out_dir, exist_ok=True)
    ref_file = os.path.join(out_dir, 'refs.pt')
    if os.path.exists(ref_file):
        refs = torch.load(ref_file, weights_only=False)
    else:
        t0 = time.perf_counter()
        refs = {'train': reference_logps(model, train_pairs, encode), 'val': reference_logps(model, val_pairs, encode)}
        refs['seconds'] = round(time.perf_counter() - t0, 1)
        trainer.Store.save(refs, ref_file)
    ref_train, ref_val = refs['train'], refs['val']
    hist = {'beta': beta, 'lr': lr, 'epochs': epochs, 'grad_accum': grad_accum, 'sft_alpha': sft_alpha,
            'n_train': len(train_pairs), 'n_val': len(val_pairs), 'ref_seconds': refs['seconds'], 'steps': [], 'epochs_log': []}
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    ctx, use_scaler = _autocast(model)
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    def step(i):
        c, r, n_c = _pair_logps(model, encode, train_pairs[i], ctx)
        rc, rr = ref_train[i]
        z = beta * ((c - rc) - (r - rr))
        loss = -F.logsigmoid(z).mean()
        if sft_alpha:
            loss = loss + sft_alpha * (-c.mean() / n_c)
        return loss, {'loss': loss.item(), 'reward_acc': float(z.item() > 0), 'margin': z.item() / beta,
                      'chosen_logratio': c.item() - rc, 'rejected_logratio': r.item() - rr}

    hist, store = trainer.fit(out_dir, len(train_pairs), epochs, step, params, opt, scaler,
                              lambda: evaluate_pairs(model, val_pairs, ref_val, encode, beta),
                              lambda: get_peft_model_state_dict(model), lambda sd: set_peft_model_state_dict(model, sd),
                              hist, model.train, grad_accum, warmup_ratio, max_grad_norm, seed, save_every, log_every, 'dpo')
    model.save_pretrained(out_dir)
    save_json(hist, os.path.join(out_dir, 'history.json'))
    store.clear()
    os.remove(ref_file)
    model.eval()
    model.config.use_cache = True
    return hist


def trained(out_dir):
    return os.path.exists(os.path.join(out_dir, 'history.json')) and os.path.exists(os.path.join(out_dir, 'adapter_config.json'))


def load_history(out_dir):
    return load_json(os.path.join(out_dir, 'history.json'))


# ----------------------------------------------------------------------------- encoders
def end_token_id(model, tok):
    e = model.generation_config.eos_token_id
    return (e[0] if isinstance(e, (list, tuple)) else e) if e is not None else tok.eos_token_id


def llm_encoder(model, tok, build_chat, max_len=640):
    """Pairs {'prompt': str, 'chosen': str, 'rejected': str} -> chat prompt + response + <|im_end|>."""
    eos = end_token_id(model, tok)
    cache = {}

    def encode(pair, which):
        key = pair['prompt']
        if key not in cache:
            cache[key] = tok(build_chat(tok, [{'role': 'user', 'content': pair['prompt']}]), add_special_tokens=False)['input_ids']
        p = cache[key]
        r = tok(pair[which], add_special_tokens=False)['input_ids'][:max(8, max_len - len(p) - 1)] + [eos]
        ids = torch.tensor([p + r], device=model.device)
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}, len(r)
    return encode


def vlm_encoder(model, proc, load_image, max_resp=192):
    """Pairs {'image': key, 'prompt': str, 'chosen': str, 'rejected': str}; the processor expands the
    image placeholder, then the response token ids are appended to every per-token tensor."""
    from .vlm import chat_text
    tok = proc.tokenizer
    eos = end_token_id(model, tok)
    dt = _base_dtype(model)

    def encode(pair, which):
        enc = proc(text=[chat_text(proc, pair['prompt'])], images=[[load_image(pair['image'])]], return_tensors='pt')
        r = tok(pair[which], add_special_tokens=False)['input_ids'][:max_resp] + [eos]
        L = enc['input_ids'].shape[1]
        out = {}
        for k, v in enc.items():
            if not torch.is_tensor(v):
                continue
            if v.dim() == 2 and v.shape == (1, L):
                fill = r if k == 'input_ids' else ([1] * len(r) if k == 'attention_mask' else [0] * len(r))
                v = torch.cat([v, torch.tensor([fill], dtype=v.dtype)], dim=1)
            v = v.to(model.device)
            out[k] = v.to(dt) if v.is_floating_point() else v
        return out, len(r)
    return encode
