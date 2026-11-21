"""
作業一 corpus and chunking (slide 8: documents -> chunks -> embeddings).

Every law is one document: its articles are concatenated as "勞動基準法第24條：<text>". Two chunking families:
  fixed(size, overlap)   character windows over each law (never across laws); a window that starts inside an
                         article gets a header "《勞動基準法》（第24條，續）" so it keeps its context (slide 12:
                         annotate every chunk with its coordinates)
  article                one chunk per article (structure-aware); articles longer than 1,000 characters are split
                         into 500-character windows with 60 characters overlap
Each chunk records which articles it covers; an article counts as covered when the chunk contains at least half of
it or 100 of its characters. Retrieval is scored against the gold articles of the evaluation questions.
"""

import json
import os

from .common import DATA_DIR

MIN_COVER = 100


def load_articles():
    with open(os.path.join(DATA_DIR, 'laws.json'), encoding='utf-8') as f:
        d = json.load(f)
    return d['articles'], d


def _law_docs(articles):
    docs = {}
    for a in articles:
        docs.setdefault(a['law'], []).append(a)
    out = []
    for law, arts in docs.items():
        text, spans = '', []
        for a in arts:
            piece = f'{a["id"]}：{a["text"]}\n'
            spans.append((a['id'], len(text), len(text) + len(piece), a['article']))
            text += piece
        out.append((law, text, spans))
    return out


def _covered(spans, s, e):
    cov = []
    for aid, a0, a1, _ in spans:
        ov = min(e, a1) - max(s, a0)
        if ov > 0 and ov >= min(MIN_COVER, (a1 - a0) / 2):
            cov.append(aid)
    return cov


def fixed_chunks(articles, size, overlap):
    step = max(1, size - overlap)
    chunks = []
    for law, text, spans in _law_docs(articles):
        s = 0
        while s < len(text):
            e = min(len(text), s + size)
            head = next(((aid, no, a0) for aid, a0, a1, no in spans if a0 <= s < a1), None)
            header = f'《{law}》' + (f'（{head[1]}，續）' if head and head[2] < s else '')
            chunks.append({'law': law, 'text': header + text[s:e], 'raw': text[s:e], 'articles': _covered(spans, s, e)})
            if e == len(text):
                break
            s += step
    return _finish(chunks)


def article_chunks(articles, max_len=1000, win=500, overlap=60):
    chunks = []
    for a in articles:
        piece = f'{a["id"]}：{a["text"]}'
        if len(piece) <= max_len:
            chunks.append({'law': a['law'], 'text': f'《{a["law"]}》' + piece, 'raw': piece, 'articles': [a['id']]})
            continue
        for s in range(0, len(piece), win - overlap):
            part = piece[s:s + win]
            chunks.append({'law': a['law'], 'text': f'《{a["law"]}》' + (part if s == 0 else f'（{a["article"]}，續）' + part),
                           'raw': part, 'articles': [a['id']]})
            if s + win >= len(piece):
                break
    return _finish(chunks)


def _finish(chunks):
    for i, c in enumerate(chunks):
        c['id'] = i
    return chunks


def make_chunks(articles, cfg):
    """cfg: {'mode': 'fixed', 'size': 256, 'overlap': 64} or {'mode': 'article'}."""
    if cfg['mode'] == 'article':
        return article_chunks(articles)
    return fixed_chunks(articles, cfg['size'], cfg['overlap'])


def cfg_name(cfg):
    return 'article' if cfg['mode'] == 'article' else f'fixed{cfg["size"]}_ov{cfg["overlap"]}'


URLS = {}


def article_url(aid, articles=None):
    if not URLS:
        for a in (articles or load_articles()[0]):
            URLS[a['id']] = a['url']
    return URLS.get(aid)
