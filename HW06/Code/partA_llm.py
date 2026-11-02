"""
Part A (作業一): open-source LLM -- deployment, custom 20-question instruction benchmark,
prompt engineering, quantization and GPU-memory measurement.

  Basic 1  deploy Qwen3-4B-Instruct-2507 (FP16) and chat with it
  Basic 2  20 self-designed instructions covering 20 abilities (Data/partA_llm_benchmark.json),
           each with a rule-based checker (quantitative metric) and a reference for the Gemini judge
  Basic 3  prompting techniques from the lecture (CoT, few-shot, emotional stimuli, XML / Markdown
           prompts, Chinese vs. English instructions) on the FP16 model
  Basic 4  precision FP16 vs INT8 (LLM.int8) vs NF4 (4-bit) with the default (baseline) prompt
  Adv. 1   average / peak GPU memory during FP16 inference (sampled every 20 ms)
  Adv. 2   best prompt x low-precision models: memory, speed and quality vs. FP16

Every generation run is cached in Output/partA_runs/, so a disconnected Colab session resumes.
"""

import os
import random
import time

import numpy as np
import pandas as pd
import torch

from .common import (DEVICE, GPUMemorySampler, load_json, save_json, out_path, free_gpu, plt, DATA_DIR,
                     gpu_info)
from .gemini import GeminiJudge, scores_by_id
from .llm import load_llm, generate_all, generate_batch, model_footprint_mib, LLM_ID
from .scoring import auto_check

MAX_NEW_TOKENS = 768
# one batch size for every configuration on a given GPU, so speed / memory numbers are comparable:
# 10 on a 16 GB T4; 5 on the 11 GB RTX 2080 Ti (FP16 weights 8 GB + KV cache of 10 long few-shot prompts would not fit)
GEN_BATCH = 10 if DEVICE.type != 'cuda' or torch.cuda.get_device_properties(0).total_memory >= 14 * 2**30 else 5
PROFILE_QIDS = ['A02', 'A08', 'A10', 'A13', 'A19']   # batch-1 memory profile (short + long answers)
PROFILE_TOKENS = 256
DEMO_PROMPTS = ['請用三句話向高中生解釋什麼是大型語言模型。', 'Write a haiku about Taipei in the rain.']

# ----------------------------------------------------------------------------- prompting techniques
FINAL_RULE = '請先一步一步思考，最後一行以「最終答案：」開頭寫出答案；若題目要求特定格式（例如 JSON、條列、詩），請在「最終答案：」之後依該格式完整輸出。'

FEW_SHOT = [
    {'role': 'user', 'content': '一本書定價 250 元，買 4 本可以打八折，總共要付多少元？'},
    {'role': 'assistant', 'content': '4 本原價 250 × 4 = 1000 元，打八折後 1000 × 0.8 = 800 元。\n最終答案：800 元'},
    {'role': 'user', 'content': '列出 2 種常見的水果。每種一行、以「- 」開頭，不要輸出任何其他文字。'},
    {'role': 'assistant', 'content': '最終答案：\n- 蘋果\n- 香蕉'},
    {'role': 'user', 'content': '請將「謝謝你的幫忙」翻譯成英文，只輸出譯文。'},
    {'role': 'assistant', 'content': '最終答案：Thank you for your help.'},
]

SYSTEM_ZH = """# 角色
你是一位嚴謹、精確的 AI 助理。

# 規則
1. 仔細閱讀問題，完全遵守題目中的所有格式、字數與語言要求。
2. 需要計算或推理的題目，先簡短地逐步思考。
3. 最後以「最終答案：」開頭寫出答案；若題目要求特定格式（JSON、條列、詩等），在「最終答案：」之後完整依該格式輸出，不要加其他文字。
4. 除非題目要求其他語言，請使用繁體中文回答。
5. 這對我的工作非常重要，請務必確認答案正確。"""

SYSTEM_EN = """# Role
You are a careful and precise AI assistant.

# Rules
1. Read the question carefully and follow every format, length and language requirement in it exactly.
2. For questions that need calculation or reasoning, think step by step briefly first.
3. End with a line starting with 「最終答案：」 followed by the answer only; if the question asks for a specific format (JSON, bullet list, poem, ...), output that content after 「最終答案：」 exactly in that format with nothing else.
4. Answer in Traditional Chinese unless the question asks for another language.
5. This is very important to my work, so make sure the answer is correct."""

TECHNIQUES = {
    'baseline':      'Zero-shot：直接輸入題目（預設 Prompt）',
    'zero_shot_cot': 'Zero-shot Chain-of-Thought：要求逐步思考並以「最終答案：」收尾',
    'few_shot':      'Few-shot：3 組示範問答（含「最終答案：」格式）',
    'emotion':       'Emotional Stimuli：加上「這對我的職涯發展非常重要」',
    'xml':           'XML 結構化 Prompt：<instruction> / <question>',
    'markdown':      'Markdown 結構化 Prompt：## 任務 / ## 問題',
    'combined_zh':   '組合（中文系統提示）：Markdown 系統提示 + XML 題目 + CoT + Few-shot + Emotional Stimuli',
    'combined_en':   '組合（英文系統提示）：同上，但系統提示改用英文',
}


def build_messages(technique, q):
    if technique == 'baseline':
        return [{'role': 'user', 'content': q}]
    if technique == 'zero_shot_cot':
        return [{'role': 'user', 'content': f'{q}\n\n{FINAL_RULE}'}]
    if technique == 'few_shot':
        return FEW_SHOT + [{'role': 'user', 'content': q}]
    if technique == 'emotion':
        return [{'role': 'user', 'content': f'{q}\n\n這對我的職涯發展非常重要，請務必認真作答。'}]
    if technique == 'xml':
        return [{'role': 'user', 'content': '<instruction>請精確回答問題，並嚴格遵守問題中的所有格式要求。</instruction>\n'
                                            f'<question>\n{q}\n</question>'}]
    if technique == 'markdown':
        return [{'role': 'user', 'content': f'## 任務\n請精確回答下列問題，並嚴格遵守所有格式要求。\n\n## 問題\n{q}'}]
    if technique in ('combined_zh', 'combined_en'):
        system = SYSTEM_ZH if technique == 'combined_zh' else SYSTEM_EN
        return [{'role': 'system', 'content': system}] + FEW_SHOT + [{'role': 'user', 'content': f'<question>\n{q}\n</question>'}]
    raise ValueError(technique)


# ----------------------------------------------------------------------------- runs
def load_benchmark():
    return load_json(os.path.join(DATA_DIR, 'partA_llm_benchmark.json'))


def run_config(model, tok, bench, precision, technique, run_dir, batch_size=GEN_BATCH, max_new_tokens=MAX_NEW_TOKENS):
    """Generate answers for all 20 questions with one (precision, technique); cached as JSON."""
    path = os.path.join(run_dir, f'{precision}__{technique}.json')
    if os.path.exists(path):
        return rescore(load_json(path), bench)
    convs = [build_messages(technique, b['question']) for b in bench]
    with GPUMemorySampler() as mem:
        res, used_bs = generate_all(model, tok, convs, batch_size, max_new_tokens)
    n_new = sum(r['n_new_tokens'] for r in res)
    run = {'precision': precision, 'technique': technique, 'batch_size': used_bs, 'max_new_tokens': max_new_tokens,
           'seconds': round(mem.seconds, 2), 'tokens_per_s': round(n_new / mem.seconds, 2), 'memory': mem.summary(),
           'answers': [{'id': b['id'], **r, 'auto': auto_check(r['text'], b['check'])} for b, r in zip(bench, res)]}
    save_json(run, path)
    print(f'  {precision:5s} {technique:14s} auto {sum(a["auto"] for a in run["answers"]):2d}/20 | '
          f'{run["tokens_per_s"]:.1f} tok/s | avg alloc {run["memory"]["avg_allocated_mib"]} MiB')
    return run


def rescore(run, bench):
    """Re-apply the current rule-based checkers to cached answers (the generations themselves are unchanged)."""
    for b, a in zip(bench, run['answers']):
        a['auto'] = auto_check(a['text'], b['check'])
    return run


def profile_memory(model, tok, bench, precision, technique, run_dir):
    """Batch-1 inference over PROFILE_QIDS: average / peak GPU memory and the memory timeline."""
    path = os.path.join(run_dir, f'profile__{precision}__{technique}.json')
    if os.path.exists(path):
        return load_json(path)
    items = [b for b in bench if b['id'] in PROFILE_QIDS]
    n_new = 0
    with GPUMemorySampler() as mem:
        for b in items:
            n_new += generate_batch(model, tok, [build_messages(technique, b['question'])], PROFILE_TOKENS)[0]['n_new_tokens']
    t = np.arange(len(mem.samples)) * mem.interval
    prof = {'precision': precision, 'technique': technique, 'n_requests': len(items), 'max_new_tokens': PROFILE_TOKENS,
            'tokens_per_s': round(n_new / mem.seconds, 2), 'memory': mem.summary(),
            'timeline': {'t': [round(x, 2) for x in t[::5]], 'allocated_mib': [round(s[0] / 2**20, 1) for s in mem.samples[::5]]}}
    save_json(prof, path)
    return prof


def pick_best_technique(runs):
    """Highest rule-based pass rate on FP16 among the non-baseline techniques (tie -> fewer tokens)."""
    cands = [(sum(a['auto'] for a in r['answers']), -sum(a['n_new_tokens'] for a in r['answers']), t)
             for (p, t), r in runs.items() if p == 'fp16' and t != 'baseline']
    return max(cands)[2]


# ----------------------------------------------------------------------------- Gemini judge
JUDGE_PROMPT = """You are an impartial grader of answers produced by AI assistants for a Chinese instruction-following benchmark.

Question:
{question}

Reference answer / grading notes:
{reference}

Below are {n} anonymous answers to the same question. Some answers show reasoning and then a line starting with 「最終答案：」; in that case grade the content after 「最終答案：」 for format compliance and do not penalise the reasoning part.
Grade each answer independently on a 1-10 scale: 10 = fully correct, follows every instruction (format, length, language) and is clear; 7 = correct but with a minor instruction violation; 4 = partly correct or a major instruction violation; 1 = wrong, fabricated or unhelpful.
Return only a JSON array with one object per answer: {{"id": "<letter>", "score": <1-10>, "reason": "<at most 15 words>"}}.

{answers}"""


def judge_all(bench, runs, judge):
    """One Gemini call per question grading every run's answer (anonymised, shuffled order)."""
    keys = list(runs)
    rows = []
    for qi, b in enumerate(bench):
        order = keys[:]
        random.Random(1000 + qi).shuffle(order)
        letters = [chr(ord('A') + i) for i in range(len(order))]
        answers = '\n\n'.join(f'[{l}]\n{runs[k]["answers"][qi]["text"]}' for l, k in zip(letters, order))
        prompt = JUDGE_PROMPT.format(question=b['question'], reference=b['reference'], n=len(order), answers=answers)
        got = scores_by_id(judge.judge(f'partA::{b["id"]}::' + '|'.join(f'{p}/{t}' for p, t in order), prompt))
        for l, k in zip(letters, order):
            g = got.get(l, {})
            rows.append({'id': b['id'], 'precision': k[0], 'technique': k[1], 'gemini_score': g.get('score'),
                         'gemini_reason': g.get('reason')})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main
def run_partA(use_gemini=True):
    run_dir = out_path('partA_runs')
    os.makedirs(run_dir, exist_ok=True)
    bench = load_benchmark()
    runs, profiles, footprints, load_times = {}, {}, {}, {}

    def load(precision):
        t0 = time.perf_counter()
        model, tok = load_llm(precision)
        load_times[precision] = round(time.perf_counter() - t0, 1)
        footprints[precision] = model_footprint_mib(model)
        print(f'[Part A] {LLM_ID} {precision}: footprint {footprints[precision]} MiB, load {load_times[precision]} s')
        return model, tok

    # FP16: all prompting techniques + batch-1 memory profile
    model, tok = load('fp16')
    demo = chat_demo(model, tok, DEMO_PROMPTS)
    demo.to_csv(out_path('partA_deploy_demo.csv'), index=False, encoding='utf-8-sig')
    for t in TECHNIQUES:
        runs[('fp16', t)] = run_config(model, tok, bench, 'fp16', t, run_dir)
    best = pick_best_technique(runs)
    print(f'[Part A] best prompting technique on FP16: {best}')
    for t in ('baseline', best):
        profiles[('fp16', t)] = profile_memory(model, tok, bench, 'fp16', t, run_dir)
    del model; free_gpu()

    # INT8 / NF4: default prompt (Basic 4) and best prompt (Advanced 2)
    if DEVICE.type == 'cuda':
        for p in ('int8', 'nf4'):
            model, tok = load(p)
            for t in ('baseline', best):
                runs[(p, t)] = run_config(model, tok, bench, p, t, run_dir)
                profiles[(p, t)] = profile_memory(model, tok, bench, p, t, run_dir)
            del model; free_gpu()

    save_json({'gpu': gpu_info(), 'model': LLM_ID, 'best_technique': best, 'footprint_mib': footprints,
               'load_seconds': load_times}, out_path('partA_meta.json'))
    return report_partA(bench, runs, profiles, footprints, best, demo, use_gemini)


def rebuild_partA_report(use_gemini=True):
    """Rebuild every Part A table / figure from the cached runs (no GPU needed), e.g. after a checker fix."""
    run_dir = out_path('partA_runs')
    bench = load_benchmark()
    meta = load_json(out_path('partA_meta.json'))
    best = meta['best_technique']
    keys = [('fp16', t) for t in TECHNIQUES] + [(p, t) for p in ('int8', 'nf4') for t in ('baseline', best)]
    runs = {k: rescore(load_json(os.path.join(run_dir, f'{k[0]}__{k[1]}.json')), bench) for k in keys
            if os.path.exists(os.path.join(run_dir, f'{k[0]}__{k[1]}.json'))}
    profiles = {k: load_json(os.path.join(run_dir, f'profile__{k[0]}__{k[1]}.json')) for k in keys
                if os.path.exists(os.path.join(run_dir, f'profile__{k[0]}__{k[1]}.json'))}
    demo = pd.read_csv(out_path('partA_deploy_demo.csv'))
    return report_partA(bench, runs, profiles, meta['footprint_mib'], best, demo, use_gemini)


def report_partA(bench, runs, profiles, footprints, best, demo, use_gemini=True):
    # answers table (+ Gemini)
    ans = pd.DataFrame([{'id': a['id'], 'ability': b['ability'], 'precision': p, 'technique': t, 'auto': a['auto'],
                         'n_new_tokens': a['n_new_tokens'], 'answer': a['text']}
                        for (p, t), r in runs.items() for b, a in zip(bench, r['answers'])])
    judge = GeminiJudge(out_path('partA_gemini_cache.json')) if use_gemini else None
    if judge is not None and (judge.available or judge.cache):
        print(f'[Part A] Gemini judge: {judge.model}')
        ans = ans.merge(judge_all(bench, runs, judge), on=['id', 'precision', 'technique'], how='left')
    else:
        ans['gemini_score'] = np.nan
        print('[Part A] no GEMINI_API_KEY -> Gemini judge skipped (rule-based metrics only)')
    ans.to_csv(out_path('partA_answers.csv'), index=False, encoding='utf-8-sig')

    def summarize(keys):
        out = []
        for k in keys:
            r, sub = runs[k], ans[(ans.precision == k[0]) & (ans.technique == k[1])]
            m = r['memory']
            out.append({'precision': k[0], 'technique': k[1], 'auto_pass': int(sub.auto.sum()),
                        'auto_pass_rate': round(sub.auto.mean(), 3),
                        'gemini_mean': round(sub.gemini_score.mean(), 2) if sub.gemini_score.notna().any() else np.nan,
                        'avg_new_tokens': round(sub.n_new_tokens.mean(), 1), 'tokens_per_s': r['tokens_per_s'],
                        'seconds': r['seconds'], 'avg_allocated_mib': m['avg_allocated_mib'],
                        'peak_allocated_mib': m['peak_allocated_mib'], 'avg_process_mib': m['avg_process_mib']})
        return pd.DataFrame(out)

    tech = summarize([('fp16', t) for t in TECHNIQUES])
    tech.insert(2, 'description', [TECHNIQUES[t] for t in tech.technique])
    quant = summarize([k for k in runs if k[1] in ('baseline', best)])
    quant.insert(2, 'footprint_mib', [footprints[p] for p in quant.precision])
    prof = pd.DataFrame([{'precision': k[0], 'technique': k[1], 'tokens_per_s_batch1': v['tokens_per_s'],
                          **{f'b1_{kk}': vv for kk, vv in v['memory'].items() if kk != 'n_samples'}}
                         for k, v in profiles.items()])
    prof.insert(2, 'footprint_mib', [footprints[p] for p in prof.precision])
    tech.to_csv(out_path('partA_technique_summary.csv'), index=False, encoding='utf-8-sig')
    quant.to_csv(out_path('partA_quant_summary.csv'), index=False, encoding='utf-8-sig')
    prof.to_csv(out_path('partA_memory_profile.csv'), index=False, encoding='utf-8-sig')
    by_ability = ans.pivot_table(index=['id'], columns=['precision', 'technique'], values='auto', aggfunc='first')
    figs = plot_partA(tech, quant, profiles, best)
    return {'demo': demo, 'answers': ans, 'techniques': tech, 'quant': quant, 'profile': prof, 'by_question': by_ability,
            'best': best, 'figures': figs}


def plot_partA(tech, quant, profiles, best):
    figs = {}
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    x = np.arange(len(tech))
    ax[0].bar(x, tech.auto_pass_rate * 100, color='#4C72B0')
    ax[0].set_xticks(x, tech.technique, rotation=30, ha='right'); ax[0].set_ylabel('Rule-based pass rate (%)')
    ax[0].set_title('Prompting techniques (FP16): rule-based metric'); ax[0].set_ylim(0, 105)
    for i, v in enumerate(tech.auto_pass_rate * 100):
        ax[0].text(i, v + 1, f'{v:.0f}', ha='center')
    if tech.gemini_mean.notna().any():
        ax[1].bar(x, tech.gemini_mean, color='#DD8452')
        for i, v in enumerate(tech.gemini_mean):
            ax[1].text(i, v + 0.1, f'{v:.2f}', ha='center')
    ax[1].set_xticks(x, tech.technique, rotation=30, ha='right'); ax[1].set_ylabel('Gemini score (1-10)')
    ax[1].set_title('Prompting techniques (FP16): Gemini judge'); ax[1].set_ylim(0, 10.5)
    fig.tight_layout(); fig.savefig(out_path('partA_techniques.png'), dpi=150); figs['techniques'] = fig

    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    short = {'baseline': 'base', 'zero_shot_cot': 'CoT', 'few_shot': 'few-shot', 'combined_zh': 'comb-zh', 'combined_en': 'comb-en'}
    labels = [f'{p}\n{short.get(t, t)}' for p, t in zip(quant.precision, quant.technique)]
    colors = ['#4C72B0' if t == 'baseline' else '#55A868' for t in quant.technique]
    for a, col, title in zip(ax, ['auto_pass_rate', 'gemini_mean', 'tokens_per_s', 'avg_allocated_mib'],
                             ['Rule-based pass rate', 'Gemini score (1-10)', f'Throughput (tok/s, batch {GEN_BATCH})',
                              f'Avg GPU memory allocated (MiB, batch {GEN_BATCH})']):
        vals = quant[col].astype(float)
        a.bar(range(len(quant)), vals, color=colors)
        a.set_xticks(range(len(quant)), labels, fontsize=8); a.set_title(title, fontsize=10)
        for i, v in enumerate(vals):
            if not np.isnan(v):
                a.text(i, v, f'{v:.2f}' if v < 20 else f'{v:.0f}', ha='center', va='bottom', fontsize=8)
    fig.suptitle(f'Precision x prompt (blue = baseline prompt, green = best prompt "{best}")')
    fig.tight_layout(); fig.savefig(out_path('partA_quantization.png'), dpi=150); figs['quantization'] = fig

    if any(v['timeline']['t'] for v in profiles.values()):
        fig, ax = plt.subplots(figsize=(10, 4))
        for (p, t), v in profiles.items():
            ax.plot(v['timeline']['t'], v['timeline']['allocated_mib'], label=f'{p} / {t}', lw=1)
        ax.set_xlabel('seconds'); ax.set_ylabel('torch.cuda.memory_allocated (MiB)')
        ax.set_title(f'GPU memory during batch-1 inference ({len(PROFILE_QIDS)} requests x {PROFILE_TOKENS} new tokens)')
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out_path('partA_memory_timeline.png'), dpi=150)
        figs['memory_timeline'] = fig
    return figs


def chat_demo(model, tok, prompts, max_new_tokens=256):
    """Basic 1: deployment demo -- answer a few prompts and report speed / memory."""
    rows = []
    for p in prompts:
        with GPUMemorySampler() as mem:
            r = generate_batch(model, tok, [[{'role': 'user', 'content': p}]], max_new_tokens)[0]
        rows.append({'prompt': p, 'answer': r['text'], 'new_tokens': r['n_new_tokens'],
                     'tokens_per_s': round(r['n_new_tokens'] / mem.seconds, 1),
                     'peak_allocated_mib': mem.summary()['peak_allocated_mib']})
    return pd.DataFrame(rows)
