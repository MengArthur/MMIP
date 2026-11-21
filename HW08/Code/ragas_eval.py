"""
作業三: RAGAS-style evaluation (Es et al., EACL 2024 demo; slides 66-69) of the 作業一 system.

Metrics (same definitions as RAGAS):
  faithfulness       answer split into atomic statements; fraction supported by the retrieved contexts
  answer_relevancy   the judge-free part of RAGAS: Qwen3-4B writes 3 questions that the answer would answer; mean
                     cosine similarity (bge-m3) to the real question; 0 for non-committal answers ("資料中沒有…")
  context_precision  each retrieved chunk judged useful / not useful for the reference answer; rank-aware average
                     precision (relevant chunks should come first)
  context_recall     reference answer split into statements; fraction attributable to the retrieved contexts
  answer_correctness F1 over statements: TP = answer statements in the reference, FP = answer statements not in it,
                     FN = reference statements missing from the answer (RAGAS factual correctness)
To stay inside the Gemini free tier, ONE judge call per answer returns everything the four LLM-judged metrics need
(statements + verdicts); the scores are computed from the verdicts in code. Judge = Gemini when a key is available,
otherwise Qwen3-4B; a subset is judged by both to report how much the two judges agree.
"""

import numpy as np
import pandas as pd

from .common import out_path
from .llm import parse_json

JUDGE = """你是 RAG 系統的評審。根據「問題」「檢索到的參考資料」「系統回答」「標準答案」完成下列判斷，只輸出 JSON。

1. answer_statements：把「系統回答」拆成獨立的事實陳述（每句只含一個事實，去掉客套話；最多 8 句）。每句判斷：
   - supported：能否由「參考資料」直接推得（true/false）
   - in_reference：是否與「標準答案」中的內容一致（true/false）
2. reference_statements：把「標準答案」拆成獨立的事實陳述（最多 6 句）。每句判斷：
   - in_context：能否由「參考資料」推得（true/false）
   - in_answer：「系統回答」是否有表達這個事實（true/false）
3. context_useful：依序判斷每一則參考資料對於得出「標準答案」是否有用（true/false 的陣列，長度等於參考資料則數）

問題：{question}

參考資料：
{contexts}

系統回答：{answer}

標準答案：{reference}

輸出格式：
{{"answer_statements": [{{"statement": "...", "supported": true, "in_reference": true}}],
  "reference_statements": [{{"statement": "...", "in_context": true, "in_answer": false}}],
  "context_useful": [true, false]}}"""

REVERSE = ('根據下面這段回答，寫出 3 個「這段回答正在回答的問題」，並判斷這段回答是否含糊不肯定（例如只說不知道、資料中沒有）。'
           '只輸出 JSON：{{"questions": ["...", "...", "..."], "noncommittal": false}}\n\n回答：{answer}')


class Judge:
    """Gemini (cached) or the local LLM; .verdicts(prompt, key) -> dict or None."""

    def __init__(self, gemini=None, llm=None):
        self.gemini = gemini if gemini is not None and (gemini.available or gemini.cache) else None
        self.llm = llm
        self.name = f'gemini ({gemini.model})' if self.gemini else 'Qwen3-4B (local)'

    def batch(self, prompts, keys):
        if self.gemini is not None:
            return [self.gemini.judge_obj(k, p) for k, p in zip(keys, prompts)]
        outs = self.llm.chat([[{'role': 'user', 'content': p}] for p in prompts], 900, desc='judge')
        return [parse_json(o, None) for o in outs]


def _contexts(rec):
    return '\n'.join(f'[{i}] {c.strip()}' for i, c in enumerate(rec['contexts'], 1)) or '（沒有檢索任何資料）'


def scores_from(v, n_ctx):
    """Verdict JSON -> the four judged metrics."""
    if not isinstance(v, dict):
        return {}
    a = [x for x in v.get('answer_statements') or [] if isinstance(x, dict)]
    r = [x for x in v.get('reference_statements') or [] if isinstance(x, dict)]
    out = {}
    if a and n_ctx:
        out['faithfulness'] = np.mean([bool(x.get('supported')) for x in a])
    if r and n_ctx:
        out['context_recall'] = np.mean([bool(x.get('in_context')) for x in r])
    useful = [bool(x) for x in (v.get('context_useful') or [])][:n_ctx]
    if n_ctx and useful:
        hits, ap = 0, 0.0
        for k, u in enumerate(useful, 1):
            if u:
                hits += 1
                ap += hits / k
        out['context_precision'] = ap / hits if hits else 0.0
    tp = sum(bool(x.get('in_reference')) for x in a)
    fp = len(a) - tp
    fn = sum(not bool(x.get('in_answer')) for x in r)
    if a or r:
        out['answer_correctness'] = tp / (tp + 0.5 * (fp + fn)) if (tp + fp + fn) else 0.0
    return out


def answer_relevancy(llm, emb, recs):
    outs = llm.chat([[{'role': 'user', 'content': REVERSE.format(answer=r['answer'])}] for r in recs], 200, desc='answer relevancy')
    vals = []
    for r, o in zip(recs, outs):
        d = parse_json(o, {}) or {}
        qs = [q for q in d.get('questions', []) if isinstance(q, str)][:3] if isinstance(d, dict) else []
        if not qs or d.get('noncommittal'):
            vals.append(0.0)
            continue
        V = emb.encode([r['question']] + qs)
        vals.append(float(np.mean(V[1:] @ V[0])))
    return vals


def evaluate(name, recs, judge, llm, emb):
    """recs: answer records with question / answer / contexts / reference. Returns per-record DataFrame."""
    prompts = [JUDGE.format(question=r.get('standalone') or r['question'], contexts=_contexts(r), answer=r['answer'],
                            reference=r['reference']) for r in recs]
    keys = [f'ragas:{name}:{r.get("id", i)}' for i, r in enumerate(recs)]
    verdicts = judge.batch(prompts, keys)
    rel = answer_relevancy(llm, emb, recs)
    rows = []
    for i, (r, v) in enumerate(zip(recs, verdicts)):
        rows.append({'system': name, 'id': r.get('id', i), 'judge_ok': isinstance(v, dict), 'answer_relevancy': rel[i],
                     **scores_from(v, len(r['contexts'])), 'answer': r['answer']})
    return pd.DataFrame(rows), verdicts


METRICS = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'answer_correctness']


def summarize(dfs):
    D = pd.concat(dfs).reindex(columns=['system', 'id'] + METRICS + ['judge_ok'])   # metrics a judge could not score stay NaN
    S = D.groupby('system', sort=False)[METRICS + ['judge_ok']].mean().reset_index()
    S.round(4).to_csv(out_path('hw3_ragas_summary.csv'), index=False)
    return S


def judge_agreement(rec_list, gemini_judge, local_judge, llm, emb, name):
    """Same answers judged by Gemini and by Qwen3-4B: mean absolute difference and correlation per metric."""
    a, _ = evaluate(name, rec_list, gemini_judge, llm, emb)
    b, _ = evaluate(name + '_local', rec_list, local_judge, llm, emb)
    rows = []
    for m in ('faithfulness', 'context_precision', 'context_recall', 'answer_correctness'):
        x, y = a[m].astype(float), b[m].astype(float)
        ok = ~(x.isna() | y.isna())
        rows.append({'metric': m, 'gemini_mean': x[ok].mean(), 'local_mean': y[ok].mean(),
                     'mean_abs_diff': (x[ok] - y[ok]).abs().mean(),
                     'pearson_r': float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 2 and x[ok].std() > 0 and y[ok].std() > 0 else np.nan})
    D = pd.DataFrame(rows)
    D.round(4).to_csv(out_path('hw3_judge_agreement.csv'), index=False)
    return D
