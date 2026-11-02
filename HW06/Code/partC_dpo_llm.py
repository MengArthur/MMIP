"""
Part C basic (作業 C): DPO fine-tuning of Qwen3-4B-Instruct-2507 (LoRA).

Optimisation aspect: when the user writes in Traditional Chinese, answer in *Taiwan* Traditional
Chinese -- no Simplified characters and no mainland-China vocabulary (軟件 -> 軟體, 視頻 -> 影片 ...).
  Basic 1  preference data: 130 self-written Traditional-Chinese prompts (Data/partC_tw_prompts.json);
           the base model answers each one, and Code/taiwan_terms.build_pair() turns the answer into a
           (chosen, rejected) pair -> 130 pairs (117 train / 13 validation)
  Basic 2  100 unrelated general-ability questions (Data/partC_general_benchmark.json):
           60 MMLU multiple-choice (57 subjects) + 40 GSM8K math word problems, in English
  Basic 3  DPO with LoRA; aspect metric on 40 held-out prompts and general accuracy, before vs. after

Two training variants with identical data and hyper-parameters (only the loss differs):
  DPO      plain DPO. Removed every mainland term but degenerated: the chosen and rejected answers differ in
           a few characters only, and DPO pushed probability mass to look-alike tokens (Simplified 体 for 體,
           Japanese kana ...) -- likelihood displacement (Razin et al., 2024)
  DPO+SFT  DPO + 1.0 x per-token NLL of the chosen answer (RPO, Pang et al., 2024; TRL rpo_alpha), which
           anchors the likelihood of the preferred answer
"""

import os
import re

import numpy as np
import pandas as pd

from .common import DATA_DIR, load_json, save_json, out_path, ckpt_path, free_gpu, plt, gpu_info
from . import dpo
from . import taiwan_terms as tw
from .llm import load_llm, generate_all, build_chat, LLM_ID

SUFFIX = '（請用 200 字以內回答）'
TW_MAX_NEW_TOKENS = 384
GEN_BATCH = 10
BASE_CFG = dict(beta=0.1, lr=1e-4, epochs=3, grad_accum=8)
# name -> (file tag, training config); the 'dpo' tag keeps the file names of the first run so its cache is reused
VARIANTS = {'DPO': ('dpo', dict(BASE_CFG, sft_alpha=0.0)),
            'DPO+SFT': ('dposft', dict(BASE_CFG, sft_alpha=1.0))}
LORA_R = 16
N_VAL = 13


def tw_prompts():
    d = load_json(os.path.join(DATA_DIR, 'partC_tw_prompts.json'))
    return [q + SUFFIX for q in d['train']], [q + SUFFIX for q in d['test']]


def general_benchmark():
    return load_json(os.path.join(DATA_DIR, 'partC_general_benchmark.json'))


def general_messages(item):
    if item['source'] == 'MMLU':
        opts = '\n'.join(f'{l}. {c}' for l, c in zip('ABCD', item['choices']))
        subj = item['subject'].replace('_', ' ')
        return [{'role': 'user', 'content': f'The following is a multiple choice question about {subj}.\n\n{item["question"]}\n{opts}\n\n'
                                            'Answer with only the letter (A, B, C, or D) of the correct option.'}]
    return [{'role': 'user', 'content': f'{item["question"]}\n\nSolve the problem step by step. End your response with a final line '
                                        "of the form 'Answer: <number>'."}]


def grade_general(item, text):
    if item['source'] == 'MMLU':
        m = re.search(r'\b([ABCD])\b', text)
        pred = m.group(1) if m else None
        return pred, int(pred == item['answer'])
    m = re.findall(r'Answer\s*[:：]\s*\$?\s*(-?[\d,]*\.?\d+)', text)
    nums = m or re.findall(r'-?\d[\d,]*\.?\d*', text)
    if not nums:
        return None, 0
    pred = nums[-1].replace(',', '').rstrip('.')
    try:
        return pred, int(abs(float(pred) - float(item['answer'])) < 1e-6)
    except ValueError:
        return pred, 0


def eval_general(model, tok, items, path):
    if os.path.exists(path):
        return load_json(path)
    out = []
    for src, max_new, bs in (('MMLU', 8, 20), ('GSM8K', 512, GEN_BATCH)):
        sub = [it for it in items if it['source'] == src]
        res, _ = generate_all(model, tok, [general_messages(it) for it in sub], bs, max_new)
        for it, r in zip(sub, res):
            pred, ok = grade_general(it, r['text'])
            out.append({'id': it['id'], 'source': src, 'answer': it['answer'], 'pred': pred, 'correct': ok, 'response': r['text']})
    save_json(out, path)
    return out


def gen_tw(model, tok, prompts, path):
    if os.path.exists(path):
        return load_json(path)
    res, _ = generate_all(model, tok, [[{'role': 'user', 'content': p}] for p in prompts], GEN_BATCH, TW_MAX_NEW_TOKENS)
    out = [{'prompt': p, 'response': r['text'], 'n_new_tokens': r['n_new_tokens'], **tw.analyse(r['text'])} for p, r in zip(prompts, res)]
    save_json(out, path)
    return out


def _has_weights(adapter_dir):
    return any(os.path.exists(os.path.join(adapter_dir, f)) for f in ('adapter_model.safetensors', 'adapter_model.bin'))


def _post_hoc_val(pm, hist, adapter_dir, val_pairs, encode, beta):
    """Adapters trained before the chosen / rejected log-ratios were logged: measure them once on the validation pairs."""
    if 'val_chosen_logratio' in hist['epochs_log'][-1] or 'post_hoc_val' in hist:
        return hist
    refs = dpo.reference_logps(pm, val_pairs, encode)
    hist['post_hoc_val'] = {k: float(v) for k, v in dpo.evaluate_pairs(pm, val_pairs, refs, encode, beta).items()}
    save_json(hist, os.path.join(adapter_dir, 'history.json'))
    return hist


def run_partC_llm():
    run_dir = out_path('partC_llm_runs')
    os.makedirs(run_dir, exist_ok=True)
    train_p, test_p = tw_prompts()
    general = general_benchmark()

    model, tok = load_llm('fp16')
    base_train = gen_tw(model, tok, train_p, os.path.join(run_dir, 'base_train_prompts.json'))
    base_test = gen_tw(model, tok, test_p, os.path.join(run_dir, 'base_test_prompts.json'))
    base_gen = eval_general(model, tok, general, os.path.join(run_dir, 'base_general.json'))

    # ---- preference dataset (Basic 1)
    pairs = [p for p in (tw.build_pair(r['prompt'], r['response']) for r in base_train) if p]
    save_json(pairs, out_path('partC_llm_dpo_pairs.json'))
    kinds = pd.Series([p['kind'] for p in pairs]).value_counts().to_dict()
    print(f'[Part C/LLM] {len(pairs)} preference pairs: {kinds}')
    if len(pairs) < 2 * N_VAL:
        raise RuntimeError(f'only {len(pairs)} preference pairs -- check the base answers in {run_dir}/base_train_prompts.json')
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(pairs))
    val_pairs, train_pairs = [pairs[i] for i in idx[:N_VAL]], [pairs[i] for i in idx[N_VAL:]]

    # ---- DPO variants (Basic 3)
    encode = dpo.llm_encoder(model, tok, build_chat)
    tests, gens, hists = {'Base': base_test}, {'Base': base_gen}, {}
    for name, (tag, cfg) in VARIANTS.items():
        adapter_dir = ckpt_path(f'partC_llm_{tag}_lora')
        test_path, gen_path = os.path.join(run_dir, f'{tag}_test_prompts.json'), os.path.join(run_dir, f'{tag}_general.json')
        if dpo.trained(adapter_dir) and _has_weights(adapter_dir):
            pm, hist = dpo.load_lora(model, adapter_dir), dpo.load_history(adapter_dir)
            print(f'[Part C/LLM] {name}: loaded trained adapter from {adapter_dir}')
            hist = _post_hoc_val(pm, hist, adapter_dir, val_pairs, encode, cfg['beta'])
        elif dpo.trained(adapter_dir) and os.path.exists(test_path) and os.path.exists(gen_path):
            pm, hist = None, dpo.load_history(adapter_dir)          # results cached, adapter weights not available
        else:
            print(f'[Part C/LLM] training {name}: {cfg}')
            pm = dpo.add_lora(model, r=LORA_R)
            hist = dpo.train_dpo(pm, train_pairs, val_pairs, encode, adapter_dir, **cfg)
        tests[name] = gen_tw(pm, tok, test_p, test_path)
        gens[name] = eval_general(pm, tok, general, gen_path)
        hists[name] = hist
        if pm is not None:
            model = pm.unload()                                       # remove the LoRA layers -> clean base model
            del pm; free_gpu()
    del model; free_gpu()

    # ---- results
    names = list(tests)
    aspect = pd.DataFrame([{'model': n, **tw.summarize([r['response'] for r in tests[n]])} for n in names])
    gen_rows = []
    for n in names:
        d = pd.DataFrame(gens[n])
        gen_rows.append({'model': n, 'MMLU_acc (60)': round(d[d.source == 'MMLU'].correct.mean(), 4),
                         'GSM8K_acc (40)': round(d[d.source == 'GSM8K'].correct.mean(), 4),
                         'overall_acc (100)': round(d.correct.mean(), 4)})
    general_tbl = pd.DataFrame(gen_rows)
    b = pd.DataFrame(base_gen)
    flips = {}
    for n in names[1:]:
        d = pd.DataFrame(gens[n])
        flips[n] = {'base_correct_now_wrong': int(((b.correct == 1) & (d.correct == 0)).sum()),
                    'base_wrong_now_correct': int(((b.correct == 0) & (d.correct == 1)).sum())}
    examples = pd.DataFrame([{'prompt': rows[0]['prompt'],
                              **{f'{n}_terms': ' '.join(r['terms']) for n, r in zip(names, rows)},
                              **{f'{n}_simplified': r['n_simplified'] for n, r in zip(names, rows)},
                              **{f'{n}_response': r['response'] for n, r in zip(names, rows)}}
                             for rows in zip(*[tests[n] for n in names])])
    aspect.to_csv(out_path('partC_llm_aspect.csv'), index=False, encoding='utf-8-sig')
    general_tbl.to_csv(out_path('partC_llm_general.csv'), index=False, encoding='utf-8-sig')
    examples.to_csv(out_path('partC_llm_test_responses.csv'), index=False, encoding='utf-8-sig')
    term_tbl = pd.DataFrame({n: pd.Series([t for r in tests[n] for t in r['terms']], dtype=object).value_counts() for n in names})
    term_tbl = term_tbl.fillna(0).astype(int).sort_values('Base', ascending=False)
    term_tbl.to_csv(out_path('partC_llm_term_counts.csv'), encoding='utf-8-sig')
    simp_tbl = pd.DataFrame({n: pd.Series([c for r in tests[n] for c in r['response'] if c in tw.SIMPLIFIED_ONLY],
                                          dtype=object).value_counts() for n in names}).fillna(0).astype(int)
    simp_tbl = simp_tbl.sort_values(names[-1], ascending=False)
    simp_tbl.to_csv(out_path('partC_llm_simplified_chars.csv'), encoding='utf-8-sig')
    training = pd.DataFrame([{'variant': n, **{k: hists[n].get(k) for k in ('beta', 'lr', 'epochs', 'sft_alpha', 'best_epoch', 'train_seconds')},
                              **{k: round(v, 4) for k, v in hists[n]['epochs_log'][-1].items() if k.startswith('val_')},
                              **{f'post_hoc_{k}': round(v, 4) for k, v in hists[n].get('post_hoc_val', {}).items()}}
                             for n in names[1:]])
    training.to_csv(out_path('partC_llm_training_summary.csv'), index=False, encoding='utf-8-sig')
    save_json({'gpu': gpu_info(), 'model': LLM_ID, 'pairs': len(pairs), 'pair_kinds': kinds, 'n_train': len(train_pairs),
               'n_val': len(val_pairs), 'variants': {n: c for n, (_, c) in VARIANTS.items()}, 'lora_r': LORA_R,
               'flips': flips}, out_path('partC_llm_meta.json'))
    return {'aspect': aspect, 'general': general_tbl, 'flips': flips, 'terms': term_tbl, 'simplified_chars': simp_tbl,
            'examples': examples, 'training': training, 'histories': hists, 'pair_kinds': kinds,
            'figures': plot_partC_llm(aspect, general_tbl, hists)}


def plot_history(hist, title, path):
    s = pd.DataFrame(hist['steps'])
    e = pd.DataFrame(hist['epochs_log'])
    steps_per_epoch = s.groupby('epoch').step.max().to_dict()
    e['step'] = e.epoch.map(lambda x: steps_per_epoch.get(x, 0))
    has_lr = 'chosen_logratio' in s
    fig, ax = plt.subplots(1, 4 if has_lr else 3, figsize=(20 if has_lr else 15, 3.8))
    ax[0].plot(s.step, s.loss, label='train (per optimiser step)'); ax[0].plot(e.step, e.val_loss, 'o-', label='validation')
    ax[0].axhline(np.log(2), ls=':', c='gray', label='ln 2 (start)'); ax[0].set_title('Training loss'); ax[0].legend(fontsize=8)
    ax[1].plot(s.step, s.reward_acc, label='train'); ax[1].plot(e.step, e.val_reward_acc, 'o-', label='validation')
    ax[1].set_title('Reward accuracy (chosen > rejected)'); ax[1].set_ylim(-0.05, 1.05); ax[1].legend(fontsize=8)
    ax[2].plot(s.step, s.margin, label='train'); ax[2].plot(e.step, e.val_margin, 'o-', label='validation')
    ax[2].set_title('Log-ratio margin (chosen - rejected)'); ax[2].legend(fontsize=8)
    if has_lr:
        ax[3].plot(s.step, s.chosen_logratio, label='chosen'); ax[3].plot(s.step, s.rejected_logratio, label='rejected')
        ax[3].axhline(0, ls=':', c='gray'); ax[3].set_title('log pi - log pi_ref (train)'); ax[3].legend(fontsize=8)
    for a in ax:
        a.set_xlabel('optimiser step')
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=150)
    return fig


def plot_partC_llm(aspect, general_tbl, hists):
    figs = {}
    for n, h in hists.items():
        tag = VARIANTS[n][0]
        figs[f'history_{tag}'] = plot_history(h, f'Part C: {n} on Qwen3-4B-Instruct-2507 (Taiwan Traditional Chinese)',
                                              out_path(f'partC_llm_training_{tag}.png'))
    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    cols = ['#4C72B0', '#C44E52', '#55A868'][:len(aspect)]
    for a, col, title in zip(ax[:3], ['clean_rate', 'with_mainland_term_rate', 'with_simplified_rate'],
                             ['Clean answers (higher = better)', 'Answers with mainland terms (lower = better)',
                              'Answers with Simplified chars (lower = better)']):
        a.bar(aspect.model, aspect[col], color=cols); a.set_title(title, fontsize=10); a.set_ylim(0, 1.05)
        for i, v in enumerate(aspect[col]):
            a.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    x = np.arange(3); w = 0.8 / len(general_tbl)
    keys = ['MMLU_acc (60)', 'GSM8K_acc (40)', 'overall_acc (100)']
    for j, (_, r) in enumerate(general_tbl.iterrows()):
        off = (j - (len(general_tbl) - 1) / 2) * w
        vals = [r[k] for k in keys]
        ax[3].bar(x + off, vals, w, label=r.model, color=cols[j])
        for xi, v in zip(x, vals):
            ax[3].text(xi + off, v, f'{v:.2f}', ha='center', va='bottom', fontsize=7)
    ax[3].set_xticks(x, ['MMLU', 'GSM8K', 'Overall']); ax[3].set_ylim(0, 1.1); ax[3].legend(fontsize=8)
    ax[3].set_title('General ability (100 unrelated questions)', fontsize=10)
    fig.tight_layout(); fig.savefig(out_path('partC_llm_results.png'), dpi=150); figs['results'] = fig
    return figs
