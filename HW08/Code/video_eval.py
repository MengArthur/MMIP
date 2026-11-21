"""
作業三 advanced: extending RAG evaluation to VideoRAG.

RAGAS only looks at text. A VideoRAG answer depends on three more things, so the evaluation adds:
  1. video retrieval       video_hit@1 / @3 (is the right video retrieved?)
  2. temporal grounding    temporal_recall: the gold time span (±5 s) is covered by a used text unit or frame;
                           frame_precision: share of the frames given to the VLM that lie in the gold span (±10 s);
                           citation_accuracy: a cited 【影片 mm:ss】 is in the right video and the gold span (±10 s)
  3. multimodal RAGAS      faithfulness and context precision judged by Gemini looking at the selected frames
                           themselves (not only text), plus answer correctness against the reference answer
  4. ablations             modality (frames + text / text only / frames only) and frame selection (VideoRAG
                           selection vs. uniform sampling), split by question type (speech vs. visual)
Gold annotations come from Data/video_qa.json, written by the student after recording the videos.
"""

import os

import numpy as np
import pandas as pd

from .common import DATA_DIR, load_json, out_path, save_json
from .video import citations, fmt

JUDGE = """你是影片問答系統的評審。系統從影片中取出了附上的 {n_img} 張畫面（依序為 {frame_list}）與下列文字資料，並給出回答。

文字資料：
{text}

問題：{question}
系統回答：{answer}
標準答案：{reference}

請判斷並只輸出 JSON：
1. answer_statements：把系統回答拆成獨立的事實陳述（最多 6 句），每句判斷 supported（能否由附上的畫面或文字資料得到）與 in_reference（是否與標準答案一致）
2. reference_statements：把標準答案拆成事實陳述（最多 5 句），每句判斷 in_context（畫面或文字資料是否包含）與 in_answer（系統回答是否有表達）
3. frame_useful：依序判斷每張畫面對回答問題是否有用（true/false 陣列，長度 {n_img}）
{{"answer_statements": [{{"statement": "...", "supported": true, "in_reference": true}}],
  "reference_statements": [{{"statement": "...", "in_context": true, "in_answer": true}}], "frame_useful": [true]}}"""

VARIANTS = {'VideoRAG (frames + text)': dict(mode='both', frames_method='videorag'),
            'text only': dict(mode='text', frames_method='videorag'),
            'frames only': dict(mode='frames', frames_method='videorag'),
            'frames + text, uniform frames': dict(mode='both', frames_method='uniform')}


def load_qa():
    p = os.path.join(DATA_DIR, 'video_qa.json')
    return load_json(p) if os.path.exists(p) else {'questions': [], 'dialogues': []}


def _in_span(t, s, e, tol):
    return s - tol <= t <= e + tol


def rule_metrics(res, gold):
    used_frames = res['frames']
    units = res['text_units']
    s, e, vid = gold['start'], gold['end'], gold['video']
    covered = any(f['video'] == vid and _in_span(f['t'], s, e, 5) for f in used_frames) or \
        any(u['video'] == vid and u['start'] <= e + 5 and u['end'] >= s - 5 for u in units)
    cites = citations(res['answer'])
    return {'video_hit@1': float(res['videos'][:1] == [vid]), 'video_hit@3': float(vid in res['videos'][:3]),
            'temporal_recall': float(covered),
            'frame_precision': np.mean([f['video'] == vid and _in_span(f['t'], s, e, 10) for f in used_frames]) if used_frames else np.nan,
            'citation_accuracy': float(any(v == vid and _in_span(t, s, e, 10) for v, t in cites)) if cites else 0.0}


def judge_one(judge, key, res, gold, rag):
    from .ragas_eval import scores_from
    imgs = [rag.image(f['file']) for f in res['frames']]
    text = '\n'.join(f'【{u["video"]} {fmt(u["start"])}】{u["text"]}' for u in res['text_units']) or '（無）'
    prompt = JUDGE.format(n_img=len(imgs), frame_list='、'.join(f'{f["video"]} {fmt(f["t"])}' for f in res['frames']) or '（無）',
                          text=text, question=res['standalone'], answer=res['answer'], reference=gold['answer'])
    v = judge.judge_obj(key, prompt, imgs or None) if judge is not None else None
    if not isinstance(v, dict):
        return {}
    sc = scores_from({**v, 'context_useful': v.get('frame_useful')}, len(imgs) + len(res['text_units']))
    sc.pop('context_precision', None)
    useful = [bool(x) for x in (v.get('frame_useful') or [])][:len(imgs)]
    if useful:
        sc['frame_usefulness'] = float(np.mean(useful))
    return sc


def evaluate(rag, judge=None):
    qa = load_qa()
    if not qa['questions']:
        print('Data/video_qa.json has no questions yet - see Data/videos/README.md')
        return None, None
    path = out_path('hw3_video_answers_v2.json')      # v2: score-fusion video retrieval
    cache = load_json(path) if os.path.exists(path) else {}
    rows = []
    for name, kw in VARIANTS.items():
        for g in qa['questions']:
            key = f'{name}|{g["id"]}'
            if key not in cache:
                cache[key] = rag.answer(g['question'], **kw)
                save_json(cache, path)
            res = cache[key]
            rows.append({'variant': name, 'id': g['id'], 'type': g.get('type', '?'), **rule_metrics(res, g),
                         **judge_one(judge, 'video:' + key, res, g, rag), 'answer': res['answer']})
    for d in qa.get('dialogues', []):
        hist = []
        for t, g in enumerate(d['turns'], 1):
            key = f'dialogue|{d["id"]}|{t}'
            if key not in cache:
                cache[key] = rag.answer(g['question'], hist)
                save_json(cache, path)
            res = cache[key]
            hist.append((g['question'], res['answer']))
            rows.append({'variant': 'multi-turn dialogue', 'id': f'{d["id"]}-{t}', 'type': g.get('type', '?'),
                         **rule_metrics(res, g), **judge_one(judge, 'video:' + key, res, g, rag), 'answer': res['answer']})
    df = pd.DataFrame(rows)
    cols = [c for c in ('video_hit@1', 'video_hit@3', 'temporal_recall', 'frame_precision', 'citation_accuracy',
                        'faithfulness', 'context_recall', 'frame_usefulness', 'answer_correctness') if c in df]
    S = df.groupby('variant', sort=False)[cols].mean().reset_index()
    T = df[df['variant'].isin(VARIANTS)].groupby(['variant', 'type'], sort=False)[[c for c in ('temporal_recall', 'answer_correctness') if c in df]].mean().reset_index()
    S.round(4).to_csv(out_path('hw3_video_summary.csv'), index=False)
    T.round(4).to_csv(out_path('hw3_video_by_type.csv'), index=False)
    df.to_csv(out_path('hw3_video_rows.csv'), index=False)
    return S, T
