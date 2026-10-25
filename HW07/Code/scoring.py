"""
Rule-based (quantitative) checkers for the Part A instruction benchmark and the Part B visual benchmark.

Each benchmark item carries a `check` spec; `auto_check(response, spec)` returns 1 (pass) or 0 (fail).
Checks are applied to the *final answer*: if the response contains a 「最終答案：」/"Final answer:"
line (requested by the chain-of-thought style prompts), only the text after the last such marker is
checked, so visible reasoning does not count against format constraints.
"""

import json
import re
import subprocess
import sys
import unicodedata

_MARKER = re.compile(r'(最終答案|最后答案|Final Answer|Final answer|final answer|FINAL ANSWER)\s*[:：]?')
_NUM = re.compile(r'-?\d+(?:,\d{3})*(?:\.\d+)?')


def normalize(text):
    """NFKC (full-width digits / ：，！？ -> ASCII), strip markdown bold and inline-code backticks."""
    t = unicodedata.normalize('NFKC', text or '')
    t = re.sub(r'\*\*|(?<!`)`(?!`)', '', t)
    return t.strip()


def final_answer(text):
    t = normalize(text)
    m = list(_MARKER.finditer(t))
    if m:
        t = t[m[-1].end():].strip()
    return t


def numbers(text):
    return [float(x.replace(',', '')) for x in _NUM.findall(text)]


def _code_block(text):
    m = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, re.S)
    if m:
        return max(m, key=len)
    return text


def _run_python(code, func, tests, timeout=10):
    harness = (code + '\n\nimport json as _j\n_r = []\nfor _a, _e in ' + repr(tests) +
               f':\n    try:\n        _r.append({func}(*_a) == _e)\n    except Exception:\n        _r.append(False)\n'
               'print(_j.dumps(_r))\n')
    try:
        p = subprocess.run([sys.executable, '-I', '-c', harness], capture_output=True, text=True, timeout=timeout)
        res = json.loads(p.stdout.strip().splitlines()[-1])
        return all(res) and len(res) == len(tests)
    except Exception:
        return False


def _lines(text):
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _cjk_len(text):
    return len(re.sub(r'\s', '', text))


def auto_check(response, spec):
    """1 if the final answer of `response` satisfies `spec`, else 0."""
    ans = final_answer(response)
    t = spec['type']
    if t == 'all':
        return int(all(auto_check(response, s) for s in spec['checks']))
    if t == 'last_number':
        ns = numbers(ans)
        return int(bool(ns) and abs(ns[-1] - spec['target']) < 1e-6)
    if t == 'any_number':
        return int(any(abs(n - spec['target']) < 1e-6 for n in numbers(ans)))
    if t == 'number_seq':
        ns, tg = numbers(ans), spec['target']
        return int(any(ns[i:i + len(tg)] == [float(x) for x in tg] for i in range(len(ns))))
    if t == 'regex':
        s = re.sub(r'\s', '', ans) if spec.get('strip_spaces') else ans
        return int(re.search(spec['pattern'], s, re.I) is not None)
    if t in ('contains_all', 'contains_any'):
        s = ans.lower() if spec.get('ignore_case') else ans
        kws = [k.lower() if spec.get('ignore_case') else k for k in spec['keywords']]
        hit = [k in s for k in kws]
        return int(all(hit) if t == 'contains_all' else any(hit))
    if t == 'label':
        return int(spec['target'] in ans and not any(o in ans for o in spec['others']))
    if t == 'last_label':       # the conclusion = the candidate mentioned last (working may mention all of them)
        pos = {c: ans.rfind(c) for c in [spec['target']] + spec['others']}
        return int(pos[spec['target']] >= 0 and pos[spec['target']] == max(pos.values()))
    if t == 'max_chars':
        body = _lines(ans)
        return int(len(body) >= 1 and _cjk_len(' '.join(body)) <= spec['n'])
    if t == 'json':
        s = re.sub(r'^```(?:json)?\s*|\s*```$', '', ans.strip())
        if not (s.startswith('{') and s.endswith('}')):
            return 0
        try:
            obj = json.loads(s)
        except json.JSONDecodeError:
            return 0
        if sorted(obj) != sorted(spec['keys']):
            return 0
        ok = all(obj.get(k) == v for k, v in spec.get('equals', {}).items())
        return int(ok and all(v in str(obj.get(k, '')) for k, v in spec.get('contains', {}).items()))
    if t == 'bullets':
        ls = _lines(ans)
        return int(len(ls) == spec['n'] and all(re.match(r'^[-－]\s*\S', ln) for ln in ls))
    if t == 'sentences':
        if any(f in ans for f in spec.get('forbid', [])):
            return 0
        body = ans.strip()
        parts = [p for p in body.split(spec['end']) if p.strip()]
        return int(body.endswith(spec['end']) and len(parts) == spec['n'])
    if t == 'lines_contain':
        ls = _lines(ans)
        return int(len(ls) == spec['n'] and all(spec['char'] in ln for ln in ls))
    if t == 'python':
        return int(_run_python(_code_block(ans), spec['func'], spec['tests']))
    # ---- Part B (English short answers from the VLMs)
    if t == 'vqa_number':
        ns = numbers(ans) or [float(w) for w in _word_numbers(ans)]
        return int(bool(ns) and abs(ns[0] - spec['target']) < 1e-6)
    if t == 'vqa_choice':
        m = re.search(r'\b([A-D])\b', ans)
        return int(m is not None and m.group(1) == spec['target'])
    if t == 'vqa_keywords':     # fraction-free: pass if at least `min_hits` of the keyword groups appear
        s = ans.lower()
        hits = sum(any(k in s for k in grp) for grp in spec['groups'])
        return int(hits >= spec.get('min_hits', len(spec['groups'])))
    if t == 'vqa_label':
        s = ans.lower()
        return int(spec['target'] in s and not any(o in s for o in spec['others']))
    if t == 'vqa_yesno':
        s = ans.strip().lower()
        y, n = re.match(r'^\W*yes\b', s) is not None, re.match(r'^\W*no\b', s) is not None
        return int((y and spec['target'] == 'yes') or (n and spec['target'] == 'no'))
    raise ValueError(f'unknown check type {t}')


_WORDS = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8,
          'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12}


def _word_numbers(text):
    return [_WORDS[w] for w in re.findall(r'[a-z]+', text.lower()) if w in _WORDS]
