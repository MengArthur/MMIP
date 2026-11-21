"""
作業一: the multi-turn RAG system (slides 8-9, 16, 23-25).

  history + question
    -> query transformation (slide 16): rewrite the last question into a standalone question using the history
       (multi-turn), optionally multi-query (3 paraphrases, fused with reciprocal rank fusion) or step-back
       (also retrieve with a more general question)
    -> dense retrieval with bge-m3 over the chunks; optionally also over synthetic questions generated for every
       chunk (slide 12: "turn the text into QA form and embed the Q") - a chunk scores the max of both
    -> optional re-ranking of the top-20 with bge-reranker-v2-m3 (two-stage retrieval, slides 25-26)
    -> answer with Qwen3-4B using one of four prompting techniques; every answer cites its sources [n], which are
       mapped back to the law articles (name, article number, URL)
"""

import re

import numpy as np

from .corpus import article_url
from .llm import parse_json

SYSTEM = '你是台灣勞動法規諮詢助理，熟悉勞動基準法等勞動法令。'

PROMPTS = {
    'basic': {
        'desc': '直接把檢索到的條文與問題交給模型',
        'system': SYSTEM,
        'user': '參考資料：\n{context}\n\n問題：{question}',
    },
    'grounded': {
        'desc': '角色 + 規則：只能依據資料回答、資料不足時說明不知道、以 [n] 標示出處、先給結論',
        'system': SYSTEM + '回答規則：\n1. 只能根據「參考資料」回答，不可使用資料以外的知識或自行推測。\n'
                  '2. 每個事實後面用 [編號] 標示出處，例如「婚假為 14 日 [2]」。\n'
                  '3. 參考資料不足以回答時，直接說明「參考資料中沒有相關規定」。\n'
                  '4. 先用一句話給出結論，再補充必要條件或例外，總長度不超過 150 字。',
        'user': '參考資料：\n{context}\n\n問題：{question}',
    },
    'cot': {
        'desc': 'grounded 的規則 + Chain-of-Thought：先逐條判斷哪些資料相關、再推理，最後輸出答案',
        'system': None,   # filled below (grounded rules + reasoning format)
        'user': '參考資料：\n{context}\n\n問題：{question}\n\n請先在「分析：」後面逐一說明哪些參考資料與問題相關、'
                '如何適用，再在「答案：」後面寫出最終回答（含 [編號] 出處）。',
    },
    'fewshot': {
        'desc': 'grounded 的規則 + 一個示範問答（格式與引用方式）',
        'system': None,
        'user': '【示範】\n參考資料：\n[1]《勞工請假規則》勞工請假規則第7條：勞工因有事故必須親自處理，得請事假，一年內合計不得超過十四日。'
                '事假期間不給工資。\n問題：事假有薪水嗎？\n回答：事假不給工資，且一年內合計不得超過 14 日 [1]。\n\n'
                '【請回答】\n參考資料：\n{context}\n\n問題：{question}\n回答：',
    },
}
PROMPTS['cot']['system'] = PROMPTS['grounded']['system']
PROMPTS['fewshot']['system'] = PROMPTS['grounded']['system']

NO_RAG = {'system': SYSTEM + '請用 150 字以內回答，先給結論。', 'user': '問題：{question}'}

REWRITE = ('以下是使用者與勞動法規助理的對話紀錄，以及使用者最新的問題。請把最新的問題改寫成「不需要看對話紀錄也能理解」的完整問題，'
           '補上省略的主詞、代名詞所指的對象與情境，適合用來搜尋法規條文。用繁體中文，只輸出改寫後的問題。\n\n對話紀錄：\n{history}\n\n最新問題：{question}')
MULTI = ('請把下面的問題改寫成 3 個意思相同、但用詞不同的問題（例如口語說法、法律用語、更具體的說法），用來搜尋台灣勞動法規，使用繁體中文。'
         '只輸出 JSON 陣列，例如 ["...", "...", "..."]。\n\n問題：{question}')
STEPBACK = ('請先「退一步」思考：下面這個具體問題屬於台灣勞動法規中的哪一個一般性主題？請寫出一個更抽象、更一般的問題'
            '（例如「勞動基準法對特別休假的規定為何？」），用繁體中文。只輸出那一個問題。\n\n具體問題：{question}')


def format_history(history, max_turns=3):
    lines = []
    for q, a in history[-max_turns:]:
        lines.append(f'使用者：{q}')
        lines.append('助理：' + re.sub(r'\s+', ' ', a)[:200])
    return '\n'.join(lines)


class Index:
    """Chunk embeddings (+ optional synthetic-question embeddings that point to their chunk)."""

    def __init__(self, chunks, E, Q=None, q2c=None):
        self.chunks, self.E = chunks, E
        self.Q, self.q2c = Q, (np.asarray(q2c) if q2c is not None else None)

    @property
    def n_embeddings(self):
        return len(self.E) + (0 if self.Q is None else len(self.Q))

    def scores(self, qv, use_questions=True):
        s = self.E @ qv
        if use_questions and self.Q is not None:
            sq = self.Q @ qv
            best = np.full(len(self.chunks), -1.0, dtype=np.float32)
            np.maximum.at(best, self.q2c, sq)
            s = np.maximum(s, best)
        return s

    def search(self, qv, k, use_questions=True):
        s = self.scores(qv, use_questions)
        top = np.argsort(-s)[:k]
        return [(int(i), float(s[i])) for i in top]


def rrf(rankings, k=60):
    """Reciprocal rank fusion of several ranked lists of chunk ids."""
    score = {}
    for r in rankings:
        for rank, cid in enumerate(r):
            score[cid] = score.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(score, key=lambda c: -score[c])


class RAG:
    def __init__(self, llm, emb, index, reranker=None, top_k=5, prompt='grounded', query_mode='rewrite',
                 use_questions=True, rerank_pool=20, max_new_tokens=400):
        self.llm, self.emb, self.index, self.reranker = llm, emb, index, reranker
        self.top_k, self.prompt, self.query_mode = top_k, prompt, query_mode
        self.use_questions, self.rerank_pool, self.max_new_tokens = use_questions, rerank_pool, max_new_tokens

    # ------------------------------------------------------------------ query transformation
    def standalone(self, items):
        """items: list of (question, history) -> standalone questions (LLM rewrite only when there is history)."""
        idx = [i for i, (q, h) in enumerate(items) if h and self.query_mode != 'none']
        out = [q for q, _ in items]
        if idx:
            msgs = [[{'role': 'user', 'content': REWRITE.format(history=format_history(items[i][1]), question=items[i][0])}] for i in idx]
            for i, r in zip(idx, self.llm.chat(msgs, 96, desc='rewrite')):
                out[i] = r.strip().split('\n')[0] or items[i][0]
        return out

    def expansions(self, queries):
        """Extra retrieval queries per query for multi-query / step-back."""
        if self.query_mode == 'multi':
            res = self.llm.chat([[{'role': 'user', 'content': MULTI.format(question=q)}] for q in queries], 200, desc='multi-query')
            return [[x for x in (parse_json(r, []) or []) if isinstance(x, str)][:3] for r in res]
        if self.query_mode == 'stepback':
            res = self.llm.chat([[{'role': 'user', 'content': STEPBACK.format(question=q)}] for q in queries], 96, desc='step-back')
            return [[r.strip().split('\n')[0]] for r in res]
        return [[] for _ in queries]

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, queries, extra=None, k=None):
        """-> per query a list of (chunk_id, score) of length k (after optional fusion and re-ranking)."""
        k = k or self.top_k
        extra = extra or [[] for _ in queries]
        flat = [q for q, ex in zip(queries, extra) for q in [q] + ex]
        V = self.emb.encode(flat)
        out, j = [], 0
        pool = max(k, self.rerank_pool) if self.reranker else k
        for q, ex in zip(queries, extra):
            lists = [[c for c, _ in self.index.search(V[j + t], pool, self.use_questions)] for t in range(1 + len(ex))]
            j += 1 + len(ex)
            cand = rrf(lists)[:pool] if len(lists) > 1 else lists[0]
            if self.reranker:
                sc = self.reranker.score(q, [self.index.chunks[c]['text'] for c in cand])
                order = np.argsort(-sc)[:k]
                out.append([(cand[i], float(sc[i])) for i in order])
            else:
                sv = self.index.scores(V[j - 1 - len(ex)], self.use_questions)
                out.append([(c, float(sv[c])) for c in cand[:k]])
        return out

    # ------------------------------------------------------------------ generation
    def messages(self, question, hits, history=None, prompt=None):
        p = PROMPTS[prompt or self.prompt]
        ctx = '\n'.join(f'[{n}] {self.index.chunks[c]["text"].strip()}' for n, (c, _) in enumerate(hits, 1))
        msgs = [{'role': 'system', 'content': p['system']}]
        for q, a in (history or [])[-2:]:          # keep the recent turns for a coherent reply
            msgs += [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': a}]
        msgs.append({'role': 'user', 'content': p['user'].format(context=ctx, question=question)})
        return msgs

    def answer_batch(self, items, prompt=None):
        """items: list of (question, history). Returns one record per item."""
        sq = self.standalone(items)
        hits = self.retrieve(sq, self.expansions(sq))
        msgs = [self.messages(q, h, hist, prompt) for (q, hist), h in zip(items, hits)]
        raw = self.llm.chat(msgs, self.max_new_tokens, desc='answer')
        recs = []
        for (q, hist), s, h, r in zip(items, sq, hits, raw):
            ans = final_answer(r)
            cited = sorted({int(n) for n in re.findall(r'\[(\d+)\]', ans) if 0 < int(n) <= len(h)})
            refs = []
            for n, (c, sc) in enumerate(h, 1):
                ch = self.index.chunks[c]
                refs.append({'n': n, 'chunk': c, 'score': round(sc, 4), 'cited': n in cited, 'articles': ch['articles'],
                             'urls': [article_url(a) for a in ch['articles']], 'text': ch['text']})
            recs.append({'question': q, 'standalone': s, 'answer': ans, 'raw': r, 'references': refs,
                         'contexts': [x['text'] for x in refs]})
        return recs

    def answer(self, question, history=None, prompt=None):
        return self.answer_batch([(question, history or [])], prompt)[0]


def final_answer(text):
    """The part after "答案：" for chain-of-thought outputs, otherwise the whole text."""
    m = re.search(r'答案[:：]\s*(.*)', text or '', re.S)
    return (m.group(1) if m else text or '').strip()


def no_rag_answers(llm, questions, max_new_tokens=400):
    msgs = [[{'role': 'system', 'content': NO_RAG['system']}, {'role': 'user', 'content': NO_RAG['user'].format(question=q)}]
            for q in questions]
    return llm.chat(msgs, max_new_tokens, desc='no-RAG')
