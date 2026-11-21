"""
作業四: GraphRAG over one AI paper (Self-RAG), following the slides' outline (49-54) and Edge et al. 2024
("From Local to Global: A Graph RAG Approach to Query-Focused Summarization").

Indexing
  1. paper -> 1,200-character text units (150 overlap); the reference list is removed
  2. every text unit -> Qwen3-4B extracts entities (name, type, description) and relations (source, target,
     relation, description) as JSON
  3. entity resolution: names are merged by a normalised form (case, spaces, hyphens; "Self-RAG" = "SELF-RAG")
  4. graph (networkx): nodes = entities (descriptions, source units), edges = relations (weight = how often seen)
  5. communities: Louvain on the weighted graph; every community with >= 3 entities gets an LLM-written report
     (title + summary), as in Microsoft GraphRAG
Querying
  local   embed the question -> most similar entities (bge-m3 over "name: description") -> their 1-hop
          neighbourhood (relations by weight) + the text units they come from -> answer
  global  map: the 12 community reports most similar to the question each answer it with a 0-100 helpfulness score; reduce: the best
          partial answers are combined into the final answer (for whole-paper questions)
  graph   (the GraphRAG answer that is evaluated) the question is routed to global search when the LLM judges that
          it needs an overview of the whole paper, otherwise to local search
  vector  baseline RAG: top-5 text units by embedding similarity
"""

import json
import os
import re

import numpy as np

from .common import DATA_DIR, load_json, out_path, save_json
from .llm import parse_json

TYPES = ['METHOD', 'MODEL', 'COMPONENT', 'TOKEN', 'DATASET', 'TASK', 'METRIC', 'RESULT', 'HYPERPARAMETER', 'CONCEPT', 'ORGANIZATION', 'PERSON']

EXTRACT = """You build a knowledge graph of a research paper. From the text below, extract
- entities: important methods, models, components, special tokens, datasets, tasks, metrics, numeric results, hyper-parameters,
  concepts, organizations and people. type must be one of: {types}. description: one sentence with the facts the text states
  about the entity (keep numbers exactly as written).
- relations between those entities: source, target, a short relation phrase, and a one-sentence description.
Use the paper's own names (e.g. "Self-RAG", "ISSUP", "PopQA"). Output JSON only:
{{"entities": [{{"name": "...", "type": "...", "description": "..."}}],
  "relations": [{{"source": "...", "target": "...", "relation": "...", "description": "..."}}]}}

Text:
{text}"""

COMMUNITY = """Below are entities and relations that form one community of a knowledge graph built from the Self-RAG paper.
Write a report about this community: a short title and a summary paragraph (at most 120 words) that states the key facts,
numbers and how the entities relate. Output JSON only: {{"title": "...", "summary": "..."}}

Entities:
{entities}

Relations:
{relations}"""

ROUTE = ('下面是關於一篇 AI 論文（Self-RAG）的問題。如果回答它需要綜觀整篇論文（例如整理全文重點、完整流程、主要結論、與其他方法的整體比較），'
         '回答 global；如果只需要論文中特定位置的事實（定義、數值、設定、某個實驗結果），回答 local。只輸出 global 或 local。\n\n問題：{q}')
ANSWER = ('你是研究論文問答助理。根據下列從 Self-RAG 論文建立的資料回答問題，只能使用資料中的資訊，數字要照資料寫；資料不足時請說明。'
          '用繁體中文回答，保留英文專有名詞，不超過 150 字。\n\n{context}\n\n問題：{q}')
MAP = ('以下是 Self-RAG 論文知識圖譜中一個社群的報告。請判斷它對回答問題的幫助程度（0–100），並根據報告寫出部分答案（沒有幫助就寫空字串）。'
       '只輸出 JSON：{{"score": 0, "answer": "..."}}\n\n報告：{report}\n\n問題：{q}')
REDUCE = ('以下是從 Self-RAG 論文不同部分整理出的部分答案（依幫助程度排序）。請整合成一個完整、不重複的回答，只能使用這些內容。'
          '用繁體中文回答，保留英文專有名詞，不超過 200 字。\n\n{parts}\n\n問題：{q}')


def salvage(text):
    """Extraction output cut off by the token limit (Qwen3-4B pretty-prints the JSON): keep every complete
    entity / relation object instead of dropping the whole text unit."""
    ents, rels = [], []
    for s in re.findall(r'\{[^{}]*\}', text or ''):
        try:
            o = json.loads(s)
        except ValueError:
            continue
        if isinstance(o, dict) and o.get('name'):
            ents.append(o)
        elif isinstance(o, dict) and o.get('source') and o.get('target'):
            rels.append(o)
    return {'entities': ents, 'relations': rels} if ents or rels else None


def norm(name):
    n = re.sub(r'[\s\-_]+', '', (name or '').lower())
    return re.sub(r'[^\w.%]', '', n)


def paper_units(size=1200, overlap=150):
    with open(os.path.join(DATA_DIR, 'selfrag.txt'), encoding='utf-8') as f:
        text = f.read()
    i, j = text.find('REFERENCES'), text.find('APPENDIX')
    if 0 < i < j:                       # drop the bibliography, keep the appendix
        text = text[:i] + text[j:]
    text = re.sub(r'[ \t]+', ' ', text)
    units, s = [], 0
    while s < len(text):
        units.append({'id': len(units), 'text': text[s:s + size]})
        s += size - overlap
    return units


class GraphIndex:
    def __init__(self, llm, emb, cache=None):
        self.llm, self.emb = llm, emb
        self.cache = cache or out_path('hw4_graph_v2.json')     # v2: truncated extraction outputs are salvaged

    # ------------------------------------------------------------------ build
    def build(self):
        if os.path.exists(self.cache):
            self._load(load_json(self.cache))
            return self
        units = paper_units()
        outs = self.llm.chat([[{'role': 'user', 'content': EXTRACT.format(types=', '.join(TYPES), text=u['text'])}]
                              for u in units], 1200, desc='graph extraction')
        nodes, edges = {}, {}
        n_bad = n_salvaged = 0
        for u, o in zip(units, outs):
            d = parse_json(o, None)
            if not isinstance(d, dict):
                d = salvage(o)
                if d is None:
                    n_bad += 1
                    continue
                n_salvaged += 1
            for e in d.get('entities') or []:
                if not isinstance(e, dict) or not e.get('name'):
                    continue
                k = norm(e['name'])
                if not k:
                    continue
                nd = nodes.setdefault(k, {'name': e['name'], 'types': {}, 'descriptions': [], 'units': set()})
                nd['types'][e.get('type', 'CONCEPT')] = nd['types'].get(e.get('type', 'CONCEPT'), 0) + 1
                if e.get('description') and e['description'] not in nd['descriptions']:
                    nd['descriptions'].append(e['description'])
                nd['units'].add(u['id'])
            for r in d.get('relations') or []:
                if not isinstance(r, dict):
                    continue
                a, b = norm(r.get('source')), norm(r.get('target'))
                if not a or not b or a == b:
                    continue
                for k, nm in ((a, r['source']), (b, r['target'])):   # endpoints that were not listed as entities
                    nodes.setdefault(k, {'name': nm, 'types': {'CONCEPT': 1}, 'descriptions': [], 'units': set()})['units'].add(u['id'])
                key = tuple(sorted((a, b)))
                ed = edges.setdefault(key, {'source': a, 'target': b, 'relations': [], 'weight': 0, 'units': set()})
                ed['weight'] += 1
                txt = f'{r.get("relation", "")}: {r.get("description", "")}'.strip(': ')
                if txt and txt not in ed['relations']:
                    ed['relations'].append(txt)
                ed['units'].add(u['id'])
        data = {'units': units, 'n_unparsed_units': n_bad, 'n_salvaged_units': n_salvaged,
                'nodes': [{'key': k, 'name': v['name'], 'type': max(v['types'], key=v['types'].get),
                           'description': ' '.join(v['descriptions'][:4]), 'units': sorted(v['units'])} for k, v in nodes.items()],
                'edges': [{**v, 'units': sorted(v['units'])} for v in edges.values()]}
        data['communities'] = self._communities(data)
        save_json(data, self.cache)
        self._load(data)
        return self

    def _graph(self, data):
        import networkx as nx
        G = nx.Graph()
        for n in data['nodes']:
            G.add_node(n['key'])
        for e in data['edges']:
            G.add_edge(e['source'], e['target'], weight=e['weight'])
        return G

    def _communities(self, data, min_size=3):
        import networkx as nx
        G = self._graph(data)
        comms = nx.community.louvain_communities(G, weight='weight', seed=42)
        comms = [sorted(c) for c in comms if len(c) >= min_size]
        by_key = {n['key']: n for n in data['nodes']}
        prompts, members = [], []
        for c in comms:
            deg = sorted(c, key=lambda k: -G.degree(k, weight='weight'))[:25]
            ents = '\n'.join(f'- {by_key[k]["name"]} ({by_key[k]["type"]}): {by_key[k]["description"][:300]}' for k in deg)
            cs = set(c)
            rels = [e for e in data['edges'] if e['source'] in cs and e['target'] in cs]
            rels = sorted(rels, key=lambda e: -e['weight'])[:30]
            rel_txt = '\n'.join(f'- {by_key[e["source"]]["name"]} -- {by_key[e["target"]]["name"]}: {e["relations"][0][:200]}' for e in rels)
            prompts.append([{'role': 'user', 'content': COMMUNITY.format(entities=ents, relations=rel_txt or '-')}])
            members.append(c)
        reports = self.llm.chat(prompts, 600, desc='community reports')
        out = []
        for i, (c, r) in enumerate(zip(members, reports)):
            d = parse_json(r, None)
            if not isinstance(d, dict):            # truncated / malformed JSON: take the fields by pattern
                t = re.search(r'"title"\s*:\s*"([^"]*)', r)
                m = re.search(r'"summary"\s*:\s*"(.*)', r, re.S)
                d = {'title': t.group(1) if t else f'community {i}',
                     'summary': re.sub(r'"\s*}?\s*(```)?\s*$', '', m.group(1)).strip() if m else r}
            out.append({'id': i, 'members': c, 'title': d.get('title') or f'community {i}', 'summary': d.get('summary') or r})
        return out

    def _load(self, data):
        self.data = data
        self.units = data['units']
        self.nodes = {n['key']: n for n in data['nodes']}
        self.keys = list(self.nodes)
        self.G = self._graph(data)
        self.edges = {tuple(sorted((e['source'], e['target']))): e for e in data['edges']}
        self.comms = data['communities']
        self.NE = self.emb.encode([f'{self.nodes[k]["name"]}: {self.nodes[k]["description"]}' for k in self.keys], name='graph_nodes')
        self.UE = self.emb.encode([u['text'] for u in self.units], name='paper_units')
        self.CE = self.emb.encode([f'{c["title"]}: {c["summary"]}' for c in self.comms], name='graph_communities')

    def stats(self):
        import networkx as nx
        from collections import Counter
        return {'text_units': len(self.units), 'truncated_but_salvaged_units': self.data.get('n_salvaged_units', 0),
                'unparsed_units': self.data['n_unparsed_units'], 'entities': self.G.number_of_nodes(),
                'relations': self.G.number_of_edges(), 'communities': len(self.comms),
                'connected_components': nx.number_connected_components(self.G),
                'entity_types': dict(Counter(n['type'] for n in self.nodes.values()).most_common())}

    # ------------------------------------------------------------------ query
    def local_context(self, q, qv, n_ent=8, n_rel=25, n_units=4):
        sims = self.NE @ qv
        top = [self.keys[i] for i in np.argsort(-sims)[:n_ent]]
        rels = []
        for k in top:
            for nb in self.G.neighbors(k):
                e = self.edges[tuple(sorted((k, nb)))]
                rels.append((e['weight'] + (nb in top), k, nb, e))
        seen, rel_lines = set(), []
        for w, a, b, e in sorted(rels, key=lambda x: -x[0]):
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            rel_lines.append(f'- {self.nodes[a]["name"]} — {self.nodes[b]["name"]}：{"; ".join(e["relations"][:2])[:300]}')
            if len(rel_lines) >= n_rel:
                break
        cnt = {}
        for k in top:
            for u in self.nodes[k]['units']:
                cnt[u] = cnt.get(u, 0) + 1
        # source text units of the selected entities, ranked by relevance to the question (+ a small bonus for units
        # shared by several of the entities), as in GraphRAG local search
        us = sorted(cnt, key=lambda u: -(float(self.UE[u] @ qv) + 0.02 * cnt[u]))[:n_units]
        ent_lines = [f'- {self.nodes[k]["name"]} ({self.nodes[k]["type"]})：{self.nodes[k]["description"][:400]}' for k in top]
        ctx = ('實體：\n' + '\n'.join(ent_lines) + '\n\n關係：\n' + '\n'.join(rel_lines) + '\n\n原文片段：\n' +
               '\n---\n'.join(self.units[u]['text'] for u in us))
        return ctx, {'entities': [self.nodes[k]['name'] for k in top], 'units': us}

    def answer(self, questions, mode='graph', max_comms=12):
        """mode: graph (routed) | local | global | vector. Returns one record per question.
        global: map over the max_comms community reports most similar to the question (dynamic community selection),
        then reduce."""
        qv = self.emb.encode(questions)
        if mode == 'graph':
            routes = self.llm.chat([[{'role': 'user', 'content': ROUTE.format(q=q)}] for q in questions], 8)
            routes = ['global' if 'global' in r.lower() else 'local' for r in routes]
        else:
            routes = [mode] * len(questions)
        recs = [None] * len(questions)
        loc = [i for i, r in enumerate(routes) if r in ('local', 'vector')]
        msgs, meta = [], []
        for i in loc:
            if routes[i] == 'local':
                ctx, m = self.local_context(questions[i], qv[i])
            else:
                us = list(np.argsort(-(self.UE @ qv[i]))[:5])
                ctx, m = '原文片段：\n' + '\n---\n'.join(self.units[u]['text'] for u in us), {'units': [int(u) for u in us]}
            msgs.append([{'role': 'user', 'content': ANSWER.format(context=ctx, q=questions[i])}])
            meta.append((ctx, m))
        for i, a, (ctx, m) in zip(loc, self.llm.chat(msgs, 400, desc=f'{mode} answers'), meta):
            recs[i] = {'question': questions[i], 'route': routes[i], 'answer': a, 'context': ctx, **m}
        glob = [i for i, r in enumerate(routes) if r == 'global']
        if glob:
            sel = {i: [self.comms[j] for j in np.argsort(-(self.CE @ qv[i]))[:max_comms]] for i in glob}
            maps = [[{'role': 'user', 'content': MAP.format(report=f'{c["title"]}：{c["summary"]}', q=questions[i])}]
                    for i in glob for c in sel[i]]
            outs = iter(self.llm.chat(maps, 300, desc='global map'))
            red, parts_all = [], []
            for i in glob:
                parts = []
                for c, o in zip(sel[i], outs):
                    d = parse_json(o, {}) or {}
                    if isinstance(d, dict) and d.get('answer') and float(d.get('score', 0) or 0) > 0:
                        parts.append((float(d['score']), d['answer'], c['id']))
                parts = sorted(parts, key=lambda x: -x[0])[:8]
                parts_all.append(parts)
                txt = '\n'.join(f'[{s:.0f}] {a}' for s, a, _ in parts) or '（沒有相關內容）'
                red.append([{'role': 'user', 'content': REDUCE.format(parts=txt, q=questions[i])}])
            for i, a, parts in zip(glob, self.llm.chat(red, 500, desc='global reduce'), parts_all):
                recs[i] = {'question': questions[i], 'route': 'global', 'answer': a,
                           'context': '\n'.join(p[1] for p in parts), 'communities': [p[2] for p in parts]}
        return recs

    def plot(self, path=None, top=60):
        import networkx as nx
        from .common import plt
        deg = sorted(self.G.degree(weight='weight'), key=lambda x: -x[1])[:top]
        H = self.G.subgraph([k for k, _ in deg])
        comm_of = {k: c['id'] for c in self.comms for k in c['members']}
        pos = nx.spring_layout(H, seed=3, k=0.6)
        fig, ax = plt.subplots(figsize=(13, 10))
        nx.draw_networkx_edges(H, pos, alpha=0.25, width=[0.5 + 0.5 * H[a][b]['weight'] for a, b in H.edges()], ax=ax)
        nx.draw_networkx_nodes(H, pos, node_size=[60 + 25 * d for _, d in H.degree(weight='weight')],
                               node_color=[comm_of.get(k, -1) for k in H.nodes()], cmap='tab20', ax=ax)
        nx.draw_networkx_labels(H, pos, {k: self.nodes[k]['name'][:22] for k in H.nodes()}, font_size=7, ax=ax)
        ax.set_title(f'Self-RAG knowledge graph: {top} most connected of {self.G.number_of_nodes()} entities (colour = community)')
        ax.axis('off'); plt.tight_layout()
        plt.savefig(path or out_path('hw4_graph.png'), dpi=130); plt.show()


# ----------------------------------------------------------------------------- evaluation
JUDGE = """你是研究論文問答的評審。根據「標準答案」與「關鍵要點」評估「系統回答」，只輸出 JSON。
- key_points：依序判斷每個關鍵要點是否被系統回答正確涵蓋（true/false，數字必須一致）
- correctness：整體正確性 1–5（5 = 完全正確且完整，1 = 錯誤或答非所問）
- has_error：系統回答是否包含與論文矛盾的錯誤陳述（true/false）

問題：{q}
標準答案：{ref}
關鍵要點：{kp}
系統回答：{ans}

輸出格式：{{"key_points": [true, false], "correctness": 4, "has_error": false}}"""


def evaluate(recs_by_mode, qa, judge_obj):
    """judge_obj(key, prompt) -> dict (Gemini or local). Key-point recall, 1-5 correctness, error rate."""
    import pandas as pd
    rows = []
    for mode, recs in recs_by_mode.items():
        for g, r in zip(qa, recs):
            p = JUDGE.format(q=g['question'], ref=g['answer'], kp='；'.join(f'({i + 1}) {k}' for i, k in enumerate(g['key_points'])),
                             ans=r['answer'])
            v = judge_obj(f'graph:{mode}:{g["id"]}', p)
            kp = [bool(x) for x in (v.get('key_points') or [])][:len(g['key_points'])] if isinstance(v, dict) else []
            rows.append({'system': mode, 'id': g['id'], 'type': g['type'], 'route': r.get('route'),
                         'key_point_recall': np.mean(kp) if kp else np.nan,
                         'correctness': float(v.get('correctness', np.nan)) if isinstance(v, dict) else np.nan,
                         'has_error': float(bool(v.get('has_error'))) if isinstance(v, dict) else np.nan, 'answer': r['answer']})
    df = pd.DataFrame(rows)
    S = df.groupby('system', sort=False)[['key_point_recall', 'correctness', 'has_error']].mean().reset_index()
    T = df.pivot_table(index='system', columns='type', values='key_point_recall', sort=False).reset_index()
    df.to_csv(out_path('hw4_rows.csv'), index=False)
    S.round(4).to_csv(out_path('hw4_summary.csv'), index=False)
    T.round(4).to_csv(out_path('hw4_by_type.csv'), index=False)
    return S, T, df
