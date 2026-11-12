"""
The four routing nodes compared in 作業二 (all built on Qwen3-VL-2B-Instruct), plus Gemini as a commercial reference.

  A  prompt only      long system prompt with the full spec, rules and an example; free generation
  B  trained          LoRA SFT on 720 dialogues (and SFT + DPO); short system prompt; free generation
  C  decoding         the prompt of A + FSM-constrained decoding (constrained.py)
  B+C                 the trained model of B + constrained decoding
  D  System One       no generation: one prefill pass + a classification head (system_one.py)
Every router returns the same record: parsed object, raw text, token counts, latency and TTFT.
"""

import json

from . import schema
from .vlm import generate

SYSTEM_FULL = f"""你是「速達 3C 線上商城」客服系統的分流節點。閱讀客服對話紀錄，判斷客戶最新的需求應該如何處理，只輸出一個 JSON 物件，不要輸出任何其他文字、說明或 Markdown 標記。

{schema.spec_text()}

{schema.RULES}

輸出格式範例（key 的順序固定為 status, department, urgency, intent, order_id, product, missing_info）：
客戶只說「我想退掉藍牙耳機」（缺訂單編號）→ {{"status":"clarify","department":"return_refund","urgency":"normal","intent":"退貨","order_id":null,"product":"藍牙耳機","missing_info":["order_id"]}}
客戶說「訂單 SD12345678 的平板電池膨脹了」（資訊齊全、有安全疑慮）→ {{"status":"route","department":"repair_warranty","urgency":"high","intent":"電池膨脹","order_id":"SD12345678","product":"平板","missing_info":[]}}"""

PROMPT_TAIL = '請依規則輸出客戶最新需求的分流 JSON。'
SYSTEM_SHORT = '你是速達 3C 客服分流器，閱讀對話紀錄後只輸出分流 JSON。'


def record(text, gen=None, strict_obj=None):
    """Parse and validate one raw output."""
    strict = schema.parse(text, strict=True) if strict_obj is None else strict_obj
    lenient = strict if strict is not None else schema.parse(text, strict=False)
    errs = schema.schema_errors(lenient) if lenient is not None else ['no json']
    rec = {'raw': text, 'json_strict': strict is not None, 'json_lenient': lenient is not None,
           'schema_valid': lenient is not None and not errs, 'schema_errors': errs,
           'rule_errors': schema.rule_errors(lenient) if lenient is not None and not errs else None, 'obj': lenient}
    if gen:
        rec.update({k: gen[k] for k in ('n_in', 'n_out', 'n_images', 'latency', 'ttft', 'truncated')})
    return rec


class GenerativeRouter:
    """A / B / C / B+C: system prompt + transcript -> generated JSON (optionally FSM-constrained)."""

    def __init__(self, model, proc, system, prompt=None, decoder=None, max_new_tokens=160):
        self.model, self.proc, self.system, self.prompt = model, proc, system, prompt
        self.decoder, self.max_new_tokens = decoder, max_new_tokens

    def __call__(self, turns):
        g = generate(self.model, self.proc, turns, self.system, self.prompt, self.max_new_tokens,
                     processor_factory=self.decoder.processor if self.decoder else None)
        rec = record(g['text'], g)
        if g['fsm'] is not None:
            rec['mask_seconds'] = g['fsm'].mask_seconds
            rec['fsm_violations'] = g['fsm'].violations
        return rec


def prompt_router(model, proc):
    return GenerativeRouter(model, proc, SYSTEM_FULL, PROMPT_TAIL)


def constrained_router(model, proc, decoder):
    return GenerativeRouter(model, proc, SYSTEM_FULL, PROMPT_TAIL, decoder)


def trained_router(model, proc, decoder=None):
    return GenerativeRouter(model, proc, SYSTEM_SHORT, None, decoder)


# ----------------------------------------------------------------------------- Gemini (commercial reference)
class GeminiRouter:
    """Same prompt as A; structured=True passes the JSON schema as response_schema (the API-side equivalent
    of experiment C, like OpenAI Structured Outputs on slide 58). Responses are cached; latency includes
    the network round trip."""

    def __init__(self, judge, structured):
        self.judge, self.structured = judge, structured

    def __call__(self, turns, key):
        from .vlm import load_image
        lines, images = [], []
        for t in turns:
            tag = '[客戶]' if t['role'] == 'user' else '[客服]'
            for rel in t['images']:
                images.append(load_image(rel))
                lines.append(f'{tag}（傳送圖片 {len(images)}）')
            if t['text']:
                lines.append(f'{tag} {t["text"]}')
        from .vlm import HEADER
        prompt = SYSTEM_FULL + '\n\n' + HEADER + '\n' + '\n'.join(lines) + '\n\n' + PROMPT_TAIL
        cache_key = ('C:' if self.structured else 'A:') + key
        r = self.judge.complete(cache_key, prompt, images, schema.JSON_SCHEMA if self.structured else None)
        if r is None:
            return None
        rec = record(r['text'])
        rec.update({'n_in': r['n_in'], 'n_out': r['n_out'], 'n_images': len(images), 'latency': r['latency'],
                    'ttft': None, 'truncated': False, 'model': r['model']})
        return rec


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False)
