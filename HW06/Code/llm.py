"""
Open-source LLM loading (FP16 / INT8 / NF4 via bitsandbytes) and batched chat generation.

Model: Qwen3-4B-Instruct-2507 (4.0 B parameters, Apache-2.0, not gated) -- the lecture's
"Qwen3-4B, FP16" recommendation for a 12-16 GB GPU (slide 17). FP16 weights are 8.0 GB, so
it also fits the 11 GB RTX 2080 Ti.
"""

import time

import torch

from .common import DEVICE, default_dtype, free_gpu, logits_finite

LLM_ID = 'Qwen/Qwen3-4B-Instruct-2507'
PRECISIONS = ('fp16', 'int8', 'nf4')


def load_llm(precision='fp16', model_id=LLM_ID):
    """precision: 'fp16' (default precision), 'int8' (LLM.int8), 'nf4' (4-bit NormalFloat + double quant)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = default_dtype()
    kw = {'dtype': dtype}
    if precision == 'int8':
        kw['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == 'nf4':
        kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                       bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    elif precision != 'fp16':
        raise ValueError(precision)
    if DEVICE.type == 'cuda':
        kw['device_map'] = {'': 0}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    probe = tok(['你好，請自我介紹。'], return_tensors='pt').to(model.device)
    if dtype == torch.float16 and not logits_finite(model, probe):
        # fp16 overflow guard: T4 / 2080 Ti emulate bf16 (slower but numerically safe)
        print(f'[llm] non-finite logits in fp16 -> reloading {precision} with bf16 compute')
        del model; free_gpu()
        kw['dtype'] = torch.bfloat16
        if 'quantization_config' in kw and precision == 'nf4':
            kw['quantization_config'].bnb_4bit_compute_dtype = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    model.generation_config.pad_token_id = tok.pad_token_id
    return model, tok


def model_footprint_mib(model):
    return round(model.get_memory_footprint() / 2**20, 1)


def build_chat(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def generate_batch(model, tok, conversations, max_new_tokens=512):
    """conversations: list of message lists. Greedy decoding (deterministic), left padding.
    Returns [{'text', 'n_prompt_tokens', 'n_new_tokens'}] and the wall time of the batch."""
    prompts = [build_chat(tok, m) for m in conversations]
    enc = tok(prompts, return_tensors='pt', padding=True, add_special_tokens=False).to(model.device)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                         top_k=None, pad_token_id=tok.pad_token_id)
    new = out[:, enc['input_ids'].shape[1]:]
    res = []
    for i in range(len(prompts)):
        ids = new[i]
        stop = (ids == tok.pad_token_id) | (ids == tok.eos_token_id)
        n = int(stop.nonzero()[0]) if stop.any() else len(ids)
        res.append({'text': tok.decode(ids[:n], skip_special_tokens=True).strip(),
                    'n_prompt_tokens': int(enc['attention_mask'][i].sum()), 'n_new_tokens': n})
    return res


def generate_all(model, tok, conversations, batch_size=10, max_new_tokens=512):
    """Batched generation over many conversations with OOM back-off. Adds per-item latency share."""
    from .common import run_batched

    def fn(chunk):
        t0 = time.perf_counter()
        r = generate_batch(model, tok, chunk, max_new_tokens)
        dt = time.perf_counter() - t0
        for x in r:
            x['batch_seconds'] = round(dt, 3)
        return r

    return run_batched(fn, conversations, batch_size)
