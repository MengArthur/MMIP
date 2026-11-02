"""
Open-source vision-language models (Part B and Part C advanced), both recommended on slide 57:
  Qwen3-VL-2B-Instruct     2.1 B  ViT encoder + MLP merger projected into the LLM (LLaVA-style, DeepStack)
  SmolVLM2-2.2B-Instruct   2.2 B  SigLIP encoder + pixel-shuffle connector + SmolLM2 (Idefics3-style)
Both run in FP16 (~4.5 GB) on a T4 or 2080 Ti.
"""

import torch
from PIL import Image

from .common import DEVICE, default_dtype, free_gpu, logits_finite

VLMS = {'Qwen3-VL-2B': 'Qwen/Qwen3-VL-2B-Instruct', 'SmolVLM2-2.2B': 'HuggingFaceTB/SmolVLM2-2.2B-Instruct'}


def load_vlm(model_id, max_pixels=None):
    """max_pixels: optional cap on Qwen-VL input resolution (fewer visual tokens -> faster DPO / POPE)."""
    from transformers import AutoModelForImageTextToText, AutoProcessor
    proc = AutoProcessor.from_pretrained(model_id)
    proc.tokenizer.padding_side = 'left'
    if max_pixels and 'qwen' in model_id.lower():
        proc.image_processor.size = {'longest_edge': max_pixels, 'shortest_edge': min(65536, max_pixels)}
    dtype = default_dtype()
    kw = {'dtype': dtype}
    if DEVICE.type == 'cuda':
        kw['device_map'] = {'': 0}
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kw).eval()
    probe = build_inputs(proc, [Image.new('RGB', (224, 224), (120, 160, 200))], ['Describe the image.']).to(model.device)
    if dtype == torch.float16 and not logits_finite(model, probe):
        print(f'[vlm] non-finite logits in fp16 -> reloading {model_id} in bf16')
        del model; free_gpu()
        kw['dtype'] = torch.bfloat16
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kw).eval()
    tok = proc.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.generation_config.pad_token_id = tok.pad_token_id
    return model, proc


def chat_text(proc, prompt, system=None):
    msgs = ([{'role': 'system', 'content': [{'type': 'text', 'text': system}]}] if system else []) + \
           [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
    return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def build_inputs(proc, images, prompts, system=None):
    """One image + one text prompt per sample; left-padded batch."""
    texts = [chat_text(proc, p, system) for p in prompts]
    enc = proc(text=texts, images=[[im.convert('RGB')] for im in images], return_tensors='pt', padding=True)
    return enc


def _to_device(enc, model):
    dt = next(model.parameters()).dtype
    out = {}
    for k, v in enc.items():
        if torch.is_tensor(v):
            v = v.to(model.device)
            if v.is_floating_point():
                v = v.to(dt)
        out[k] = v
    return out


@torch.no_grad()
def generate(model, proc, images, prompts, max_new_tokens=128, system=None):
    enc = _to_device(build_inputs(proc, images, prompts, system), model)
    tok = proc.tokenizer
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None,
                         top_k=None, pad_token_id=tok.pad_token_id)
    new = out[:, enc['input_ids'].shape[1]:]
    res = []
    for ids in new:
        stop_ids = {tok.pad_token_id, tok.eos_token_id} | set(_eos_ids(model))
        n = next((i for i, t in enumerate(ids.tolist()) if t in stop_ids), len(ids))
        res.append({'text': tok.decode(ids[:n], skip_special_tokens=True).strip(), 'n_new_tokens': n})
    return res


def _eos_ids(model):
    e = model.generation_config.eos_token_id
    return e if isinstance(e, (list, tuple)) else ([e] if e is not None else [])


def yes_no_ids(tok):
    """First token ids of 'Yes' / 'No' (and lower-case variants) for logit scoring."""
    ids = {}
    for word in ('Yes', 'No', 'yes', 'no'):
        t = tok.encode(word, add_special_tokens=False)
        ids[word] = t[0]
    return [ids['Yes'], ids['yes']], [ids['No'], ids['no']]


@torch.no_grad()
def score_yes_no(model, proc, images, questions, suffix=' Answer with Yes or No.'):
    """P(yes) from the next-token logits after the generation prompt (POPE-style probing)."""
    enc = _to_device(build_inputs(proc, images, [q + suffix for q in questions]), model)
    logits = model(**enc).logits[:, -1].float()
    yes, no = yes_no_ids(proc.tokenizer)
    ly = torch.logsumexp(logits[:, yes], dim=-1)
    ln = torch.logsumexp(logits[:, no], dim=-1)
    return torch.sigmoid(ly - ln).cpu().tolist()
