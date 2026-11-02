"""
Part B (作業 B): two open-source VLMs on a self-designed 20-question visual benchmark.

  Basic 1  deploy Qwen3-VL-2B-Instruct and SmolVLM2-2.2B-Instruct (FP16)
  Basic 2  20 visual-understanding questions (Data/partB_vlm_benchmark.json): classification, counting,
           scene-text OCR, attributes, spatial relations, multi-object detection, scene reasoning,
           fine-grained recognition, hallucination probe, image description, and 5 synthetic images
           (sign OCR, shape counting, bar chart, spatial layout, visual math) whose answers are exact
  Basic 3  same prompt for both models; quantitative metric = rule-based accuracy per question type,
           plus a commercial LLM judge (Gemini sees the image, the reference and both anonymised answers)
"""

import os
import random
import textwrap

import numpy as np
import pandas as pd
from PIL import Image

from .common import GPUMemorySampler, DATA_DIR, load_json, save_json, out_path, free_gpu, plt, gpu_info
from .gemini import GeminiJudge, scores_by_id
from .scoring import auto_check
from .vlm import VLMS, load_vlm, generate

MAX_NEW_TOKENS = 128


def load_benchmark():
    items = load_json(os.path.join(DATA_DIR, 'partB_vlm_benchmark.json'))
    for it in items:
        it['pil'] = Image.open(os.path.join(DATA_DIR, 'partB_images', it['image'])).convert('RGB')
    return items


def run_model(name, items, run_dir):
    path = os.path.join(run_dir, f'{name}.json')
    if os.path.exists(path):
        return load_json(path)
    model, proc = load_vlm(VLMS[name])
    footprint = round(model.get_memory_footprint() / 2**20, 1)
    answers = []
    with GPUMemorySampler() as mem_all:
        for it in items:                      # batch 1: per-question latency is comparable across models
            with GPUMemorySampler(interval=0.05) as mem:
                r = generate(model, proc, [it['pil']], [it['question']], MAX_NEW_TOKENS)[0]
            answers.append({'id': it['id'], 'text': r['text'], 'n_new_tokens': r['n_new_tokens'],
                            'seconds': round(mem.seconds, 3), 'auto': auto_check(r['text'], it['check'])})
    n_img_tokens = int(proc(text=[proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'x'}]}],
                                                          add_generation_prompt=True, tokenize=False)],
                            images=[[items[0]['pil']]], return_tensors='pt')['input_ids'].shape[1])
    run = {'model': name, 'hf_id': VLMS[name], 'footprint_mib': footprint, 'memory': mem_all.summary(),
           'prompt_tokens_640x480': n_img_tokens, 'answers': answers}
    save_json(run, path)
    del model; free_gpu()
    print(f'[Part B] {name}: auto {sum(a["auto"] for a in answers)}/20, footprint {footprint} MiB')
    return run


JUDGE_PROMPT = """You are an impartial grader of answers produced by vision-language models. Look at the image.

Question: {question}
Reference answer (written by a human who checked the image): {reference}

Grade each anonymous answer independently on a 1-10 scale: 10 = fully correct and follows the requested answer format; 7 = correct but imprecise or not in the requested format; 4 = partly correct; 1 = wrong or describes things that are not in the image.
Return only a JSON array: [{{"id": "<letter>", "score": <1-10>, "reason": "<at most 15 words>"}}, ...].

{answers}"""


def judge(items, runs, judge_obj):
    names = list(runs)
    rows = []
    for qi, it in enumerate(items):
        order = names[:]
        random.Random(2000 + qi).shuffle(order)
        letters = [chr(ord('A') + i) for i in range(len(order))]
        answers = '\n'.join(f'[{l}] {runs[n]["answers"][qi]["text"]}' for l, n in zip(letters, order))
        got = scores_by_id(judge_obj.judge(f'partB::{it["id"]}::' + '|'.join(order),
                                           JUDGE_PROMPT.format(question=it['question'], reference=it['reference'], answers=answers),
                                           image=it['pil']))
        for l, n in zip(letters, order):
            g = got.get(l, {})
            rows.append({'id': it['id'], 'model': n, 'gemini_score': g.get('score'), 'gemini_reason': g.get('reason')})
    return pd.DataFrame(rows)


def run_partB(use_gemini=True):
    run_dir = out_path('partB_runs')
    os.makedirs(run_dir, exist_ok=True)
    items = load_benchmark()
    runs = {n: run_model(n, items, run_dir) for n in VLMS}
    ans = pd.DataFrame([{'id': it['id'], 'task': it['task'], 'source': it['source'], 'question': it['question'],
                         'reference': it['reference'], 'model': n, 'answer': a['text'], 'auto': a['auto'],
                         'seconds': a['seconds'], 'n_new_tokens': a['n_new_tokens']}
                        for n, r in runs.items() for it, a in zip(items, r['answers'])])
    jd = GeminiJudge(out_path('partB_gemini_cache.json')) if use_gemini else None
    if jd is not None and (jd.available or jd.cache):
        print(f'[Part B] Gemini judge: {jd.model}')
        ans = ans.merge(judge(items, runs, jd), on=['id', 'model'], how='left')
    else:
        ans['gemini_score'] = np.nan
        print('[Part B] no GEMINI_API_KEY -> Gemini judge skipped (rule-based metrics only)')
    ans.to_csv(out_path('partB_answers.csv'), index=False, encoding='utf-8-sig')

    summ = []
    for n, r in runs.items():
        sub = ans[ans.model == n]
        summ.append({'model': n, 'hf_id': r['hf_id'], 'auto_acc': round(sub.auto.mean(), 3), 'auto_pass': int(sub.auto.sum()),
                     'auto_acc_coco': round(sub[sub.source != 'synthetic'].auto.mean(), 3),
                     'auto_acc_synthetic': round(sub[sub.source == 'synthetic'].auto.mean(), 3),
                     'gemini_mean': round(sub.gemini_score.mean(), 2) if sub.gemini_score.notna().any() else np.nan,
                     'avg_latency_s': round(sub.seconds.mean(), 3), 'avg_new_tokens': round(sub.n_new_tokens.mean(), 1),
                     'prompt_tokens_640x480': r['prompt_tokens_640x480'], 'footprint_mib': r['footprint_mib'],
                     'avg_allocated_mib': r['memory']['avg_allocated_mib'], 'peak_allocated_mib': r['memory']['peak_allocated_mib']})
    summ = pd.DataFrame(summ)
    summ.to_csv(out_path('partB_summary.csv'), index=False, encoding='utf-8-sig')
    by_task = ans.pivot_table(index=['id', 'task'], columns='model', values=['auto', 'gemini_score'], aggfunc='first')
    by_task.to_csv(out_path('partB_by_question.csv'), encoding='utf-8-sig')
    save_json({'gpu': gpu_info(), 'models': VLMS}, out_path('partB_meta.json'))
    return {'answers': ans, 'summary': summ, 'by_question': by_task, 'figures': plot_partB(items, ans, summ)}


def plot_partB(items, ans, summ):
    figs = {}
    names = list(summ.model)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    cols = ['#4C72B0', '#DD8452']
    for a, col, title in zip(ax, ['auto_acc', 'gemini_mean', 'avg_latency_s'],
                             ['Rule-based accuracy (20 questions)', 'Gemini score (1-10)', 'Avg latency per question (s, batch 1)']):
        vals = summ[col].astype(float)
        a.bar(names, vals, color=cols)
        for i, v in enumerate(vals):
            if not np.isnan(v):
                a.text(i, v, f'{v:.2f}', ha='center', va='bottom')
        a.set_title(title)
    fig.tight_layout(); fig.savefig(out_path('partB_summary.png'), dpi=150); figs['summary'] = fig

    fig, axes = plt.subplots(5, 4, figsize=(20, 27))
    for ax, it in zip(axes.ravel(), items):
        ax.imshow(it['pil']); ax.axis('off')
        lines = [f'{it["id"]} ({it["source"]})', textwrap.fill('Q: ' + it['question'], 48)]
        for n in names:
            r = ans[(ans.id == it['id']) & (ans.model == n)].iloc[0]
            mark = 'O' if r.auto else 'X'
            g = '' if pd.isna(r.gemini_score) else f' | G{int(r.gemini_score)}'
            lines.append(textwrap.fill(f'{mark}{g} {n}: {textwrap.shorten(r.answer, 150)}', 48))
        ax.set_title('\n'.join(lines), fontsize=8, loc='left')
    fig.tight_layout(); fig.savefig(out_path('partB_examples.png'), dpi=110); figs['examples'] = fig
    return figs
