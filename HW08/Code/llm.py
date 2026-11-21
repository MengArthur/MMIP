"""
Local LLM: Qwen3-4B-Instruct-2507 (FP16, 8 GB; fits a T4 together with the embedding model and the re-ranker).
Used for answering (作業一), query rewriting / multi-query / step-back, synthetic questions (doc2query),
graph extraction and community summaries (作業四), and as the fallback judge when no Gemini key is available.

All generation is greedy and goes through a disk cache keyed by the full prompt, so an interrupted Colab run
continues where it stopped and every rerun reproduces the same text.
"""

import hashlib
import json
import os
import re
import time

import torch

from .common import DEVICE, default_dtype, free_gpu, logits_finite, run_batched

LLM_ID = 'Qwen/Qwen3-4B-Instruct-2507'


def load_llm(model_id=LLM_ID):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = {'dtype': default_dtype()}
    if DEVICE.type == 'cuda':
        kw['device_map'] = {'': 0}
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    probe = tok(['你好，請自我介紹。'], return_tensors='pt').to(model.device)
    if kw['dtype'] == torch.float16 and not logits_finite(model, probe):
        print('[llm] non-finite logits in fp16 -> reloading in bf16')
        del model; free_gpu()
        kw['dtype'] = torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    model.generation_config.pad_token_id = tok.pad_token_id
    return model, tok


def build_chat(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def _generate_batch(model, tok, prompts, max_new_tokens):
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
                    'n_in': int(enc['attention_mask'][i].sum()), 'n_out': n})
    return res


class LLM:
    """Qwen3-4B + prompt-keyed disk cache. chat(list of message lists) -> list of output strings."""

    def __init__(self, cache_path, model=None, tok=None, batch_size=8):
        self.cache_path = cache_path
        self.cache = {}
        if os.path.exists(cache_path):
            with open(cache_path, encoding='utf-8') as f:
                self.cache = json.load(f)
        self.model, self.tok, self.batch_size = model, tok, batch_size
        self.n_calls = 0
        self.seconds = 0.0

    def ensure_loaded(self):
        if self.model is None:
            self.model, self.tok = load_llm()

    def _save(self):
        tmp = self.cache_path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(self.cache, f, ensure_ascii=False)
        os.replace(tmp, self.cache_path)

    @staticmethod
    def key(messages, max_new_tokens):
        return hashlib.sha1(json.dumps([messages, max_new_tokens], ensure_ascii=False).encode()).hexdigest()

    def chat(self, conversations, max_new_tokens=512, desc=None):
        keys = [self.key(m, max_new_tokens) for m in conversations]
        seen = set()
        todo = [i for i, k in enumerate(keys) if k not in self.cache and not (k in seen or seen.add(k))]
        if todo:
            self.ensure_loaded()
            # longest prompts first: similar lengths share a batch (less padding)
            order = sorted(todo, key=lambda i: -len(json.dumps(conversations[i], ensure_ascii=False)))
            done = 0
            t0 = time.perf_counter()
            for s in range(0, len(order), self.batch_size * 4):
                part = order[s:s + self.batch_size * 4]
                prompts = [build_chat(self.tok, conversations[i]) for i in part]
                res, bs = run_batched(lambda ps: _generate_batch(self.model, self.tok, ps, max_new_tokens), prompts, self.batch_size)
                self.batch_size = bs        # remember a batch size that fits instead of hitting OOM again every chunk
                for i, r in zip(part, res):
                    self.cache[keys[i]] = r
                done += len(part)
                self._save()
                if desc and len(order) > self.batch_size * 4:
                    print(f'  [{desc}] {done}/{len(order)} generated  {time.perf_counter() - t0:.0f}s')
            self.seconds += time.perf_counter() - t0
            self.n_calls += len(todo)
        return [self.cache[k]['text'] for k in keys]

    def unload(self):
        """Free the GPU (the video section needs the memory); the model is reloaded on the next uncached call."""
        self.model = self.tok = None
        free_gpu()

    def one(self, messages, max_new_tokens=512):
        return self.chat([messages], max_new_tokens)[0]


def parse_json(text, default=None):
    """First JSON object / array in a model output (tolerates ```json fences and surrounding text)."""
    t = re.sub(r'^```(?:json)?|```$', '', (text or '').strip(), flags=re.M).strip()
    for cand in (t, *re.findall(r'(\{.*\}|\[.*\])', t, re.S)):
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, TypeError):
            continue
    return default
