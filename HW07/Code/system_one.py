"""
Experiment D: a "System One" routing model (lecture section 5; TypeSafe Jev, slides 61-64; architecture of slide 70).

It does not generate text. The conversation (state) goes through the Qwen3-VL-2B backbone once (prefill only,
LoRA r=16 trained), and a Joint Schema Head answers several typed questions in parallel:
  Option embeddings   every option of every question ("status: route - ...") is encoded once by the frozen
                      backbone (mean-pooled hidden states) and projected; plus a learned per-option offset
  Evidence routing    the option vectors (and one token per question) cross-attend over the state's hidden states
  Joint attention     self-attention among the options of the same question (block-diagonal mask)
  Shared scorer       one MLP gives a logit per option -> softmax per question = probabilities + confidence
Questions: status (4 choices), department (8), urgency (3) and, for each missing-information slot, a yes/no
("Noul") question - the decision fields of the routing spec. Free-text fields (intent, order id, product) are not
System One's job; the chatbot takes them from the generative router when a ticket is written.
The confidence lets the caller choose a path (slide 62): act automatically, fall back to the slower router, ask.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import dpo, schema, trainer
from .common import DEVICE, save_json
from .vlm import encode, to_device

YESNO = {'yes': '是，目前話題還缺少這項資訊，需要向客戶追問', 'no': '否，已經提供或這個需求不需要'}
QUESTIONS = [('status', '客戶最新的需求應該如何處理？', schema.STATUS),
             ('department', '應該轉給哪個部門處理？', schema.DEPARTMENTS),
             ('urgency', '這個需求的緊急程度？', schema.URGENCY)] + \
            [(f'miss_{k}', f'是否還缺少「{v}」？', YESNO) for k, v in schema.MISSING.items()]
PROMPT = '請判斷客戶最新的需求。'


def option_texts():
    texts, qidx = [], []
    for qi, (name, question, opts) in enumerate(QUESTIONS):
        for o, desc in opts.items():
            texts.append(f'問題：{question}\n選項 {o}：{desc}')
            qidx.append(qi)
    return texts, qidx


def labels_of(label):
    out = []
    for name, _, opts in QUESTIONS:
        keys = list(opts)
        if name.startswith('miss_'):
            out.append(keys.index('yes' if name[5:] in label['missing_info'] else 'no'))
        else:
            out.append(keys.index(label[name]))
    return out


def backbone(model):
    """The Qwen3-VL model below the LM head (works for a plain or a PEFT-wrapped model)."""
    m = model.get_base_model() if hasattr(model, 'get_base_model') else model
    return m.model


class JointSchemaHead(nn.Module):
    """use_state=True (v2): the hidden state of the last position (the start of the assistant turn, where a generative
    model would begin its answer, so it summarises the whole conversation) is projected into a state vector and added
    to every question / option token ("State + Schema Tokens", slide 70). v1 (use_state=False) had to find all evidence
    through cross-attention alone and did not learn the department within 3 epochs (README, D-1)."""

    def __init__(self, hidden, opt_emb, qidx, d=256, heads=4, use_state=True):
        super().__init__()
        n_q, N = len(QUESTIONS), len(qidx)
        self.n_q, self.use_state = n_q, use_state
        if use_state:
            self.state = nn.Sequential(nn.Linear(hidden, d), nn.LayerNorm(d))
        self.register_buffer('opt_emb', opt_emb.float())
        self.register_buffer('qidx', torch.tensor(qidx))
        self.ctx = nn.Sequential(nn.Linear(hidden, d), nn.LayerNorm(d))
        self.opt = nn.Sequential(nn.Linear(hidden, d), nn.LayerNorm(d))
        self.opt_bias = nn.Parameter(torch.zeros(N, d))
        self.q_tok = nn.Parameter(torch.randn(n_q, d) * 0.02)
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.joint = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.score = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        group = torch.cat([torch.arange(n_q), torch.tensor(qidx)])               # question of every token
        self.register_buffer('block', group[:, None] != group[None, :])           # True = may not attend

    def forward(self, h):
        """h: (1, L, hidden) state hidden states -> list of logit vectors, one per question."""
        c = self.ctx(h.float())
        o = self.opt(self.opt_emb) + self.opt_bias
        seq = torch.cat([self.q_tok, o], 0).unsqueeze(0)
        if self.use_state:
            seq = seq + self.state(h[:, -1].float()).unsqueeze(1)
        ev, _ = self.cross(seq, c, c)
        x = self.joint(seq + ev, src_mask=self.block)
        s = self.score(x[0, self.n_q:]).squeeze(-1)
        return [s[self.qidx == qi] for qi in range(self.n_q)]


@torch.no_grad()
def option_embeddings(model, tok):
    """Mean-pooled last hidden states of the option texts from the backbone without LoRA."""
    texts, qidx = option_texts()
    ctx = model.disable_adapter() if hasattr(model, 'disable_adapter') else torch.no_grad()
    embs = []
    with ctx:
        for t in texts:
            ids = tok(t, return_tensors='pt').to(model.device)
            h = backbone(model)(input_ids=ids['input_ids'], attention_mask=ids['attention_mask']).last_hidden_state
            embs.append(h[0].float().mean(0))
    return torch.stack(embs), qidx


class SystemOne(nn.Module):
    def __init__(self, model, proc, head, adapter=None):
        super().__init__()
        self.model, self.proc, self.head = model, proc, head
        self.adapter = adapter          # name of the LoRA adapter when it shares the base model with others

    def state(self, turns):
        enc = encode(self.proc, turns, None, PROMPT)
        return to_device(enc, self.model)

    def logits(self, enc, ctx):
        with ctx:
            h = backbone(self.model)(**enc).last_hidden_state
        with torch.autocast(DEVICE.type, enabled=False):
            return self.head(h)

    @torch.no_grad()
    def predict(self, turns):
        ctx, _ = dpo._autocast(self.model)
        self.model.eval(); self.head.eval()
        if self.adapter:
            self.model.set_adapter(self.adapter)
        enc = self.state(turns)
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        lg = self.logits(enc, ctx)
        probs = [F.softmax(x, -1).cpu().numpy() for x in lg]
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
        lat = time.perf_counter() - t0
        return decode(probs, n_in=enc['input_ids'].shape[1], latency=lat)


def decode(probs, n_in=None, latency=None):
    """Probabilities -> routing decision. The code applies the spec's consistency rules (System One only
    answers questions; what happens next is decided by the caller)."""
    raw, conf, dist = {}, {}, {}
    missing = []
    for (name, _, opts), p in zip(QUESTIONS, probs):
        keys = list(opts)
        dist[name] = {k: round(float(x), 4) for k, x in zip(keys, p)}
        if name.startswith('miss_'):
            if p[keys.index('yes')] > 0.5:
                missing.append(name[5:])
            conf[name] = float(max(p))
        else:
            raw[name] = keys[int(np.argmax(p))]
            conf[name] = float(max(p))
    raw['missing_info'] = missing
    obj = dict(raw)
    if obj['status'] in ('off_topic', 'reject'):
        obj.update(department='none', urgency='low', missing_info=[])
    elif obj['status'] == 'route':
        obj['missing_info'] = []
        if obj['department'] == 'none':           # a ticket needs a department: best one other than none
            d = dist['department']
            obj['department'] = max((k for k in d if k != 'none'), key=d.get)
    return {'obj': obj, 'raw_obj': raw, 'confidence': conf, 'probs': dist, 'n_in': n_in, 'n_out': 0,
            'latency': latency, 'ttft': latency}


# ----------------------------------------------------------------------------- training
def build(model, proc, d=256, use_state=True):
    emb, qidx = option_embeddings(model, proc.tokenizer)
    head = JointSchemaHead(emb.shape[1], emb, qidx, d=d, use_state=use_state).to(model.device)
    return SystemOne(model, proc, head)


def _loss(so, row, ctx):
    lg = so.logits(so.state(row['turns']), ctx)
    y = labels_of(row['label'])
    return sum(F.cross_entropy(x.unsqueeze(0), torch.tensor([t], device=x.device)) for x, t in zip(lg, y))


@torch.no_grad()
def evaluate(so, rows):
    ctx, _ = dpo._autocast(so.model)
    so.model.eval(); so.head.eval()
    losses, correct = [], []
    for r in rows:
        losses.append(_loss(so, r, ctx).item())
        lg = so.logits(so.state(r['turns']), ctx)
        correct.append([int(x.argmax()) == t for x, t in zip(lg, labels_of(r['label']))])
    c = np.array(correct)
    return {'loss': float(np.mean(losses)), 'acc_all_questions': float(c.all(1).mean()),
            **{f'acc_{q[0]}': float(c[:, i].mean()) for i, q in enumerate(QUESTIONS)}}


def train(so, train_rows, val_rows, out_dir, epochs=3, lr_lora=1e-4, lr_head=1e-3, grad_accum=8, warmup_ratio=0.05,
          max_grad_norm=1.0, seed=42, log_every=10, save_every=25, patience=None):
    """Trains LoRA + head together; resumable (trainer.fit). Keeps the epoch with the lowest validation loss."""
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    hist = {'epochs': epochs, 'patience': patience, 'lr_lora': lr_lora, 'lr_head': lr_head, 'grad_accum': grad_accum, 'n_train': len(train_rows),
            'n_val': len(val_rows), 'questions': [q[0] for q in QUESTIONS], 'steps': [], 'epochs_log': []}
    lora = [p for p in so.model.parameters() if p.requires_grad]
    head = list(so.head.parameters())
    opt = torch.optim.AdamW([{'params': lora, 'lr': lr_lora}, {'params': head, 'lr': lr_head}], weight_decay=0.01)
    ctx, use_scaler = dpo._autocast(so.model)
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    def step(i):
        loss = _loss(so, train_rows[i], ctx)
        return loss, {'loss': loss.item()}

    def get_state():
        return {**{'lora.' + k: v for k, v in get_peft_model_state_dict(so.model).items()},
                **{'head.' + k: v for k, v in so.head.state_dict().items()}}

    def set_state(sd):
        set_peft_model_state_dict(so.model, {k[5:]: v for k, v in sd.items() if k.startswith('lora.')})
        so.head.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith('head.')})

    def train_mode():
        so.model.train(); so.head.train()

    hist, store = trainer.fit(out_dir, len(train_rows), epochs, step, lora + head, opt, scaler, lambda: evaluate(so, val_rows),
                              get_state, set_state, hist, train_mode, grad_accum, warmup_ratio, max_grad_norm, seed,
                              save_every, log_every, 'system one', patience)
    so.model.save_pretrained(out_dir)
    torch.save(so.head.state_dict(), os.path.join(out_dir, 'head.pt'))
    save_json(hist, os.path.join(out_dir, 'history.json'))
    store.clear()
    so.model.eval(); so.head.eval()
    return hist


def load(model, proc, out_dir, d=256):
    """model: base Qwen3-VL (no adapter). Returns a SystemOne with the trained LoRA + head."""
    pm = dpo.load_lora(model, out_dir)
    sd = torch.load(os.path.join(out_dir, 'head.pt'), map_location=model.device)
    head = JointSchemaHead(sd['opt_emb'].shape[1], sd['opt_emb'], sd['qidx'].tolist(), d=d,
                           use_state=any(k.startswith('state.') for k in sd)).to(model.device)
    head.load_state_dict(sd)
    return SystemOne(pm, proc, head.eval())


def attach(pm, proc, out_dir, name='system_one', d=256):
    """Loads the System One adapter + head into an existing PeftModel (next to e.g. the SFT adapter)."""
    pm.load_adapter(out_dir, adapter_name=name)
    sd = torch.load(os.path.join(out_dir, 'head.pt'), map_location=pm.device)
    head = JointSchemaHead(sd['opt_emb'].shape[1], sd['opt_emb'], sd['qidx'].tolist(), d=d,
                           use_state=any(k.startswith('state.') for k in sd)).to(pm.device)
    head.load_state_dict(sd)
    return SystemOne(pm, proc, head.eval(), adapter=name)


def trained(out_dir):
    return all(os.path.exists(os.path.join(out_dir, f)) for f in ('history.json', 'head.pt', 'adapter_config.json'))
