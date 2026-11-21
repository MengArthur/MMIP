"""
作業一 experiments: QA dataset + embeddings, chunk size / overlap / top-k search, query transformations,
re-ranker / synthetic questions ablation, prompting techniques, multi-turn transcripts.
"""

import os

import numpy as np
import pandas as pd

from .common import load_json, out_path, plt, save_json
from .corpus import cfg_name, load_articles, make_chunks
from .qa_data import DIALOGUES, SINGLE
from .rag import RAG, Index

GRID = [{'mode': 'fixed', 'size': s, 'overlap': int(s * r)} for s in (128, 256, 512, 1024) for r in (0, 0.125, 0.25)] + \
       [{'mode': 'article'}]
KS = (1, 3, 5, 10)
CONTEXT_BUDGET = 3000          # characters of retrieved text the 4 B model gets at most
DOC2QUERY = ('以下是一段台灣勞動法規條文。請站在一般勞工或人資的角度，提出 3 個這段條文可以回答的問題，盡量使用不同的 5W1H 面向'
             '（誰、什麼、何時、多久、多少、怎麼做），其中至少一個用口語說法，3 個問題不可重複，使用繁體中文。只輸出 JSON 陣列，例如 ["...", "...", "..."]。\n\n條文：\n{chunk}')


# ----------------------------------------------------------------------------- retrieval metrics
def rank_metrics(index, emb, queries, golds, use_questions=True, reranker=None, pool=20):
    """For every query: rank (1-based) of the first chunk that covers a gold article, and recall@k of gold articles."""
    V = emb.encode(queries)
    rows = []
    for q, v, gold in zip(queries, V, golds):
        if reranker is None:
            ranked = [c for c, _ in index.search(v, max(KS), use_questions)]
        else:
            cand = [c for c, _ in index.search(v, pool, use_questions)]
            sc = reranker.score(q, [index.chunks[c]['text'] for c in cand])
            ranked = [cand[i] for i in np.argsort(-sc)][:max(KS)]
        first = next((r + 1 for r, c in enumerate(ranked) if set(index.chunks[c]['articles']) & set(gold)), None)
        row = {'rank': first, 'rr': 1.0 / first if first else 0.0}
        for k in KS:
            cov = set(a for c in ranked[:k] for a in index.chunks[c]['articles'])
            row[f'hit@{k}'] = float(first is not None and first <= k)
            row[f'recall@{k}'] = len(cov & set(gold)) / len(gold)
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_low(x, n=1000, seed=0):
    """2.5th percentile of the bootstrap mean (a pessimistic estimate: how good the setting is on a bad draw)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    return float(np.percentile(rng.choice(x, (n, len(x))).mean(1), 2.5))


def grid_search(emb, articles):
    """All chunking settings x top-k on the 47 questions, in their formal and colloquial wording."""
    qs = [q['question'] for q in SINGLE] + [q['paraphrase'] for q in SINGLE]
    golds = [q['gold'] for q in SINGLE] * 2
    style = ['formal'] * len(SINGLE) + ['colloquial'] * len(SINGLE)
    rows, per_query = [], {}
    for cfg in GRID:
        chunks = make_chunks(articles, cfg)
        E = emb.encode([c['text'] for c in chunks], name=cfg_name(cfg))
        df = rank_metrics(Index(chunks, E), emb, qs, golds, use_questions=False)
        df['style'] = style
        per_query[cfg_name(cfg)] = df
        mean_len = float(np.mean([len(c['text']) for c in chunks]))
        for k in KS:
            f, c = df[df['style'] == 'formal'], df[df['style'] == 'colloquial']
            rows.append({'config': cfg_name(cfg), 'size': cfg.get('size', 'article'), 'overlap': cfg.get('overlap', '-'),
                         'chunks': len(chunks), 'mean_chunk_chars': round(mean_len), 'k': k,
                         'context_chars': round(k * mean_len),
                         'hit_formal': f[f'hit@{k}'].mean(), 'hit_colloquial': c[f'hit@{k}'].mean(),
                         'recall': df[f'recall@{k}'].mean(), 'recall_low95': bootstrap_low(df[f'recall@{k}']),
                         'style_gap': abs(f[f'recall@{k}'].mean() - c[f'recall@{k}'].mean()),
                         'mrr': df['rr'].mean()})
    G = pd.DataFrame(rows)
    G.round(4).to_csv(out_path('hw1_grid.csv'), index=False)
    return G, per_query


def pick_best(G):
    """Most stable setting = highest pessimistic recall (bootstrap 2.5th percentile over both wordings) among the
    settings whose retrieved context fits the budget; ties -> smaller formal/colloquial gap, then shorter context."""
    ok = G[G['context_chars'] <= CONTEXT_BUDGET].sort_values(['recall_low95', 'style_gap', 'context_chars'],
                                                             ascending=[False, True, True])
    return ok.iloc[0]


def plot_grid(G, best):
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
    for cfg, g in G.groupby('config', sort=False):
        ax[0].plot(g['k'], g['recall'], 'o-', lw=1, ms=3, label=cfg)
    ax[0].set_xlabel('top-k'); ax[0].set_ylabel('recall of gold articles'); ax[0].set_xticks(KS)
    ax[0].legend(fontsize=6, ncol=2); ax[0].set_title('Recall@k by chunking setting')
    g5 = G[G['k'] == 5]
    piv = g5[g5['size'] != 'article'].pivot(index='size', columns='overlap', values='recall_low95')
    im = ax[1].imshow(piv.values, cmap='viridis', aspect='auto')
    ax[1].set_xticks(range(piv.shape[1])); ax[1].set_xticklabels([f'{c}' for c in piv.columns])
    ax[1].set_yticks(range(piv.shape[0])); ax[1].set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax[1].text(j, i, f'{v:.2f}', ha='center', va='center', color='w', fontsize=8)
    ax[1].set_xlabel('overlap (characters)'); ax[1].set_ylabel('chunk size (characters)')
    ax[1].set_title('pessimistic recall@5 (bootstrap 2.5%)'); plt.colorbar(im, ax=ax[1])
    ax[2].scatter(G['context_chars'], G['recall_low95'], s=12, alpha=0.6)
    ax[2].scatter([best['context_chars']], [best['recall_low95']], s=90, c='r', marker='*', label=f'chosen: {best["config"]}, k={best["k"]}')
    ax[2].axvline(CONTEXT_BUDGET, ls='--', c='gray', lw=1)
    ax[2].set_xscale('log'); ax[2].set_xlabel('retrieved context (characters, log)'); ax[2].set_ylabel('pessimistic recall')
    ax[2].legend(fontsize=8); ax[2].set_title('stability vs. context length')
    plt.tight_layout(); plt.savefig(out_path('hw1_grid.png'), dpi=130); plt.show()


# ----------------------------------------------------------------------------- synthetic QA dataset
def doc2query(llm, chunks):
    """3 synthetic questions per chunk (the auto-generated part of the QA dataset)."""
    from .llm import parse_json
    msgs = [[{'role': 'user', 'content': DOC2QUERY.format(chunk=c['raw'][:1200])}] for c in chunks]
    res = llm.chat(msgs, 200, desc='doc2query')
    qa = []
    for c, r in zip(chunks, res):
        qs = parse_json(r, []) or []
        for q in dict.fromkeys(q.strip() for q in (qs if isinstance(qs, list) else []) if isinstance(q, str)):   # unique
            if 4 <= len(q) <= 120:
                qa.append({'question': q, 'chunk': c['id'], 'articles': c['articles']})
    return qa


def build_index(emb, llm, articles, cfg, with_questions=True):
    chunks = make_chunks(articles, cfg)
    E = emb.encode([c['text'] for c in chunks], name=cfg_name(cfg))
    if not with_questions:
        return Index(chunks, E), []
    qa = doc2query(llm, chunks)
    Q = emb.encode([x['question'] for x in qa], name='q_' + cfg_name(cfg))
    return Index(chunks, E, Q, [x['chunk'] for x in qa]), qa


def save_dataset(index, qa):
    data = {'n_chunks': len(index.chunks), 'n_synthetic_questions': len(qa), 'n_handwritten_questions': len(SINGLE),
            'n_dialogue_turns': sum(len(d['turns']) for d in DIALOGUES), 'n_embeddings': index.n_embeddings,
            'synthetic': qa, 'handwritten': SINGLE, 'dialogues': DIALOGUES}
    save_json(data, out_path('hw1_qa_dataset.json'))
    return {k: v for k, v in data.items() if k.startswith('n_')}


# ----------------------------------------------------------------------------- ablations on the chosen setting
def ablation(index, emb, reranker):
    qs = [q['question'] for q in SINGLE] + [q['paraphrase'] for q in SINGLE]
    golds = [q['gold'] for q in SINGLE] * 2
    rows = []
    for name, uq, rr in (('chunks only', False, None), ('+ synthetic questions', True, None),
                         ('+ re-ranker', False, reranker), ('+ both', True, reranker)):
        df = rank_metrics(index, emb, qs, golds, use_questions=uq, reranker=rr)
        rows.append({'retrieval': name, **{f'hit@{k}': df[f'hit@{k}'].mean() for k in KS},
                     **{f'recall@{k}': df[f'recall@{k}'].mean() for k in KS}, 'mrr': df['rr'].mean()})
    A = pd.DataFrame(rows)
    A.round(4).to_csv(out_path('hw1_ablation.csv'), index=False)
    return A


def query_transform_eval(rag, modes=('none', 'rewrite', 'multi', 'stepback')):
    """Retrieval on the 33 dialogue turns: the history is the gold conversation (reference answers), so every mode
    sees the same context. 'none' searches with the raw last question."""
    items, golds, turn_no = [], [], []
    for d in DIALOGUES:
        hist = []
        for t, turn in enumerate(d['turns'], 1):
            items.append((turn['question'], list(hist)))
            golds.append(turn['gold'])
            turn_no.append(t)
            hist.append((turn['question'], turn['answer']))
    rows, saved = [], {}
    keep = rag.query_mode
    for mode in modes:
        rag.query_mode = mode
        sq = rag.standalone(items) if mode != 'none' else [q for q, _ in items]
        hits = rag.retrieve(sq, rag.expansions(sq) if mode in ('multi', 'stepback') else None)
        ok = [bool(set(a for c, _ in h for a in rag.index.chunks[c]['articles']) & set(g)) for h, g in zip(hits, golds)]
        saved[mode] = sq
        df = pd.DataFrame({'turn': turn_no, 'hit': ok})
        rows.append({'query_mode': mode, f'hit@{rag.top_k} all turns': df['hit'].mean(),
                     'turn 1': df[df.turn == 1]['hit'].mean(), 'turns 2-3 (need context)': df[df.turn > 1]['hit'].mean()})
    rag.query_mode = keep
    T = pd.DataFrame(rows)
    T.round(4).to_csv(out_path('hw1_query_transform.csv'), index=False)
    save_json({'items': [q for q, _ in items], 'standalone': saved}, out_path('hw1_rewrites.json'))
    return T


# ----------------------------------------------------------------------------- answers
def retrieval_tag(rag):
    """Answer files are named after the retrieval setting, so changing it never reuses answers of another setting."""
    return f'k{rag.top_k}_{"q" if rag.use_questions else "noq"}_{"rr" if rag.reranker is not None else "norr"}'


def answer_single(rag, prompts=('basic', 'grounded', 'cot', 'fewshot')):
    """Answers to the 47 questions with every prompting technique (+ the LLM without retrieval)."""
    from .rag import no_rag_answers
    out = {}
    for p in prompts:
        path = out_path(f'hw1_answers_{p}_{retrieval_tag(rag)}.json')
        if os.path.exists(path):
            out[p] = load_json(path)
            continue
        recs = rag.answer_batch([(q['question'], []) for q in SINGLE], prompt=p)
        for r, q in zip(recs, SINGLE):
            r.update(id=q['id'], reference=q['answer'], gold=q['gold'])
        save_json(recs, path)
        out[p] = recs
    path = out_path('hw1_answers_no_rag.json')
    if not os.path.exists(path):
        ans = no_rag_answers(rag.llm, [q['question'] for q in SINGLE])
        save_json([{'id': q['id'], 'question': q['question'], 'answer': a, 'contexts': [], 'reference': q['answer'],
                    'gold': q['gold'], 'references': []} for q, a in zip(SINGLE, ans)], path)
    out['no_rag'] = load_json(path)
    return out


def run_dialogues(rag, prompt=None):
    """The 11 dialogues through the full system; turn t uses the system's own earlier answers as history."""
    path = out_path(f'hw1_dialogues_{retrieval_tag(rag)}.json')
    if os.path.exists(path):
        return load_json(path)
    hist = {d['id']: [] for d in DIALOGUES}
    res = {d['id']: [] for d in DIALOGUES}
    for t in range(max(len(d['turns']) for d in DIALOGUES)):
        ds = [d for d in DIALOGUES if t < len(d['turns'])]
        recs = rag.answer_batch([(d['turns'][t]['question'], hist[d['id']]) for d in ds], prompt)
        for d, r in zip(ds, recs):
            turn = d['turns'][t]
            arts = set(a for x in r['references'] for a in x['articles'])
            cited = set(a for x in r['references'] if x['cited'] for a in x['articles'])
            r.update(turn=t + 1, reference=turn['answer'], gold=turn['gold'], retrieved_gold=bool(arts & set(turn['gold'])),
                     cited_gold=bool(cited & set(turn['gold'])))
            res[d['id']].append(r)
            hist[d['id']].append((turn['question'], r['answer']))
    save_json(res, path)
    return res


def show_dialogue(res, did):
    for r in res[did]:
        print(f'【第 {r["turn"]} 輪】使用者：{r["question"]}')
        if r['standalone'] != r['question']:
            print(f'   改寫後的檢索問題：{r["standalone"]}')
        print(f'   助理：{r["answer"]}')
        for x in r['references']:
            mark = '★' if x['cited'] else ' '
            print(f'   {mark}[{x["n"]}] {", ".join(x["articles"]) or "(條文片段)"}  score={x["score"]}')
        print(f'   標準答案：{r["reference"]}  （檢索到正確條文：{r["retrieved_gold"]}）')
