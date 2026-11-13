"""
作業二: build the four routing nodes and compare them on the 52-case benchmark (structure correctness, accuracy,
time, cost). Every stage caches its predictions in Output/, so an interrupted Colab run continues where it stopped.

Stages (each loads a fresh copy of Qwen3-VL-2B, because merging / adding adapters changes the model in memory):
  A, C          base model, long prompt; free vs FSM-constrained decoding
  B-SFT, B+C    LoRA SFT on train.json; free vs constrained decoding
  B-DPO         SFT adapter merged, new LoRA trained with DPO (reference = the SFT model)
  D             System One (backbone + LoRA + Joint Schema Head)
  Gemini-A/C    gemini-3.5-flash-lite, prompt only vs structured output (optional, needs GEMINI_API_KEY)
"""

import math
import os
import time

import numpy as np
import pandas as pd

from . import dpo, schema, sft, system_one
from .common import CKPT_DIR, DATA_DIR, DEVICE, GPUMemorySampler, free_gpu, load_json, out_path, plt, save_json, set_seed
from .constrained import ConstrainedDecoder
from .routers import GeminiRouter, constrained_router, prompt_router, trained_router
from .vlm import eos_ids, load_vlm

GPU_PRICE_PER_HOUR = 0.35   # USD, on-demand NVIDIA T4 on Google Cloud (reference price; Colab free tier costs 0)
SFT_DIR = os.path.join(CKPT_DIR, 'sft_lora')
DPO_DIR = os.path.join(CKPT_DIR, 'sft_dpo_lora')
S1_DIR_V1 = os.path.join(CKPT_DIR, 'system_one')        # v1: cross-attention only (first run, did not learn)
S1_DIR_V2 = os.path.join(CKPT_DIR, 'system_one_v2')     # v2: + state token, 3 epochs (second run)
S1_DIR_LONG = os.path.join(CKPT_DIR, 'system_one_v2_long')   # v2 trained for up to 10 epochs with early stopping (third run)
S1_DIR = S1_DIR_V2   # System One used by the chatbot: the 3-epoch v2, whose high-confidence predictions are reliable (README 2-3)
S1_LONG_EPOCHS, S1_PATIENCE = 10, 2
METHODS = ['A_prompt', 'C_constrained', 'B_sft', 'B_sft_dpo', 'B+C_sft_constrained', 'B+C_sft_dpo_constrained',
           'D_system_one', 'D_system_one_v2', 'D_system_one_v2_long', 'Gemini_prompt', 'Gemini_structured']
LABEL = {'A_prompt': 'A Prompt', 'C_constrained': 'C Constrained decoding', 'B_sft': 'B SFT', 'B_sft_dpo': 'B SFT+DPO',
         'B+C_sft_constrained': 'B+C SFT + constrained', 'B+C_sft_dpo_constrained': 'B+C SFT+DPO + constrained',
         'D_system_one': 'D System One v1', 'D_system_one_v2': 'D System One v2 (+state, 3 ep)',
         'D_system_one_v2_long': 'D System One v2 (+state, 10 ep)',
         'Gemini_prompt': 'Gemini prompt', 'Gemini_structured': 'Gemini structured output'}
TYPED = ('D_system_one', 'D_system_one_v2', 'D_system_one_v2_long')            # typed output: no JSON parsing, decision fields only


def data(name):
    return load_json(os.path.join(DATA_DIR, f'{name}.json'))


def pred_path(method):
    return out_path(f'pred_{method}.json')


def done(method):
    return os.path.exists(pred_path(method))


def run(method, router, rows, warmup=2):
    """Runs router(turns) on every row (after `warmup` untimed calls) while sampling GPU memory. Every finished
    case is appended to pred_<method>.partial.json, so a disconnected run only redoes the unfinished cases."""
    part = out_path(f'pred_{method}.partial.json')
    prev = load_json(part) if os.path.exists(part) else {'records': [], 'memory_segments': []}
    have = {r['id'] for r in prev['records']}
    todo = [r for r in rows if r['id'] not in have]
    if have:
        print(f'[{method}] resuming: {len(have)} cases already done, {len(todo)} to go')
    if todo:
        for r in todo[:warmup]:
            router(r['turns'])
        with GPUMemorySampler() as mem:
            for r in todo:
                rec = router(r['turns'])
                rec.update(id=r['id'], category=r['category'])
                prev['records'].append(_jsonable(rec))
                save_json(prev, part)
        prev['memory_segments'].append(mem.summary())
    pos = {r['id']: i for i, r in enumerate(rows)}
    out = {'method': method, 'memory': _merge_memory(prev['memory_segments']),
           'records': sorted(prev['records'], key=lambda r: pos[r['id']])}
    save_json(out, pred_path(method))
    os.remove(part)
    print(f'[{method}] done: {len(out["records"])} cases, peak GPU {out["memory"]["peak_allocated_mib"]} MiB')
    return out


def _merge_memory(segs):
    """GPU-memory summaries of the sessions a resumed run was split into: peaks = max, averages weighted by samples."""
    segs = [m for m in segs if m and m.get('n_samples')]
    if not segs:
        return {'peak_allocated_mib': None, 'avg_allocated_mib': None, 'n_samples': 0}
    w = np.array([m['n_samples'] for m in segs], float)
    out = {'n_samples': int(w.sum()), 'sessions': len(segs), 'seconds': round(sum(m['seconds'] for m in segs), 2)}
    for k in ('avg_allocated_mib', 'avg_reserved_mib', 'avg_process_mib'):
        v = [m.get(k) for m in segs]
        out[k] = round(float(np.average([x for x in v if x is not None], weights=w[[x is not None for x in v]])), 1)             if any(x is not None for x in v) else None
    for k in ('peak_allocated_mib', 'peak_reserved_mib'):
        out[k] = max(m.get(k) or 0 for m in segs)
    return out


def _jsonable(rec):
    return {k: v for k, v in rec.items() if k not in ('fsm',)}


# ----------------------------------------------------------------------------- stages
def stage_prompt_and_constrained(bench):
    if done('A_prompt') and done('C_constrained'):
        return
    model, proc = load_vlm()
    decoder = ConstrainedDecoder(proc.tokenizer, model.config.text_config.vocab_size, model.device, eos_ids(model, proc.tokenizer))
    if not done('A_prompt'):
        run('A_prompt', prompt_router(model, proc), bench)
    if not done('C_constrained'):
        run('C_constrained', constrained_router(model, proc, decoder), bench)
    del model; free_gpu()


def stage_sft(bench, train, val, epochs=2):
    if done('B_sft') and done('B+C_sft_constrained'):
        return
    set_seed(42)
    model, proc = load_vlm()
    if dpo.trained(SFT_DIR):
        pm = dpo.load_lora(model, SFT_DIR)
    else:
        pm = dpo.add_lora(model, r=16, alpha=32)
        sft.train_sft(pm, proc, train, val, SFT_DIR, epochs=epochs)
    decoder = ConstrainedDecoder(proc.tokenizer, model.config.text_config.vocab_size, model.device, eos_ids(model, proc.tokenizer))
    if not done('B_sft'):
        run('B_sft', trained_router(pm, proc), bench)
    if not done('B+C_sft_constrained'):
        run('B+C_sft_constrained', trained_router(pm, proc, decoder), bench)
    del pm, model; free_gpu()


def merged_sft():
    model, proc = load_vlm()
    pm = dpo.load_lora(model, SFT_DIR)
    return pm.merge_and_unload().eval(), proc


def stage_dpo(bench, train, val, n_train=360, n_val=60, epochs=2):
    if done('B_sft_dpo') and done('B+C_sft_dpo_constrained'):
        return
    set_seed(42)
    model, proc = merged_sft()
    if dpo.trained(DPO_DIR):
        pm = dpo.load_lora(model, DPO_DIR)
    else:
        tr, va = sft.make_pairs(train, n_train, seed=0), sft.make_pairs(val, n_val, seed=1)
        save_json({'train': tr, 'val': va}, out_path('dpo_pairs.json'))
        pm = dpo.add_lora(model, r=16, alpha=32)
        dpo.train_dpo(pm, tr, va, sft.dialogue_encoder(pm, proc), DPO_DIR, beta=0.1, lr=5e-5, epochs=epochs,
                      grad_accum=8, sft_alpha=1.0)
    if not done('B_sft_dpo'):
        run('B_sft_dpo', trained_router(pm, proc), bench)
    if not done('B+C_sft_dpo_constrained'):   # DPO invented department names (README 2-3): does the FSM fix it?
        decoder = ConstrainedDecoder(proc.tokenizer, model.config.text_config.vocab_size, model.device, eos_ids(model, proc.tokenizer))
        run('B+C_sft_dpo_constrained', trained_router(pm, proc, decoder), bench)
    del pm, model; free_gpu()


def stage_system_one(bench, train, val, version='v2'):
    """v1: cross-attention only, 3 epochs; v2: + state token, 3 epochs; v2_long: v2 trained until the validation loss
    stops improving (at most 10 epochs, patience 2; the best epoch is kept)."""
    method, out_dir = {'v1': ('D_system_one', S1_DIR_V1), 'v2': ('D_system_one_v2', S1_DIR_V2),
                       'v2_long': ('D_system_one_v2_long', S1_DIR_LONG)}[version]
    epochs, patience = (S1_LONG_EPOCHS, S1_PATIENCE) if version == 'v2_long' else (3, None)
    if done(method):
        return
    set_seed(42)
    model, proc = load_vlm()
    if system_one.trained(out_dir):
        so = system_one.load(model, proc, out_dir)
    else:
        pm = dpo.add_lora(model, r=16, alpha=32)
        so = system_one.build(pm, proc, use_state=version != 'v1')
        system_one.train(so, train, val, out_dir, epochs=epochs, patience=patience)
        pm.config.use_cache = True
    run(method, so.predict, bench)
    del so, model; free_gpu()


def stage_gemini(bench):
    from .gemini import GeminiJudge
    judge = GeminiJudge(out_path('gemini_router_cache.json'))
    for method, structured in (('Gemini_prompt', False), ('Gemini_structured', True)):
        if done(method):
            continue
        router = GeminiRouter(judge, structured)
        recs = []
        for r in bench:
            rec = router(r['turns'], r['id'])
            if rec is None:
                print(f'[{method}] skipped: no GEMINI_API_KEY and no cached responses')
                recs = None
                break
            rec.update(id=r['id'], category=r['category'])
            recs.append(rec)
        if recs:
            save_json({'method': method, 'memory': None, 'model': judge.model, 'records': recs}, pred_path(method))
            print(f'[{method}] done ({judge.model})')


# ----------------------------------------------------------------------------- metrics
def score_method(method, bench):
    P = load_json(pred_path(method))
    gold = {r['id']: r['label'] for r in bench}
    typed = method in TYPED
    fields = schema.DECISION if typed else schema.SCORED
    rows = []
    for rec in P['records']:
        s = schema.score(rec['obj'], gold[rec['id']], fields)
        if typed:
            s['exact'] = all(s[f] for f in schema.DECISION)
        else:
            s['exact'] = all(s[f] for f in schema.SCORED)
        rows.append({'id': rec['id'], 'category': rec['category'], **s,
                     'json_strict': rec.get('json_strict', np.nan), 'schema_valid': rec.get('schema_valid', np.nan),
                     'rule_ok': (rec.get('rule_errors') == []) if not typed else np.nan,
                     'latency': rec['latency'], 'ttft': rec.get('ttft'), 'n_in': rec.get('n_in'), 'n_out': rec.get('n_out')})
    return pd.DataFrame(rows), P


def train_seconds(method):
    h = {'B_sft': SFT_DIR, 'B+C_sft_constrained': SFT_DIR, 'B_sft_dpo': DPO_DIR, 'B+C_sft_dpo_constrained': DPO_DIR,
         'D_system_one': S1_DIR_V1, 'D_system_one_v2': S1_DIR_V2, 'D_system_one_v2_long': S1_DIR_LONG}.get(method)
    if not h or not os.path.exists(os.path.join(h, 'history.json')):
        return 0.0
    s = load_json(os.path.join(h, 'history.json')).get('train_seconds', 0.0)
    if method in ('B_sft_dpo', 'B+C_sft_dpo_constrained'):
        s += load_json(os.path.join(SFT_DIR, 'history.json')).get('train_seconds', 0.0)
    return s


def summary(bench):
    from .gemini import cost_usd
    rows, per_cat = [], []
    for m in METHODS:
        if not done(m):
            continue
        df, P = score_method(m, bench)
        lat = df['latency'].astype(float)
        if m.startswith('Gemini'):
            usd = np.mean([cost_usd(P['model'], r['n_in'], r['n_out']) for r in P['records']])
        else:
            usd = lat.mean() / 3600 * GPU_PRICE_PER_HOUR
        tsec = train_seconds(m)
        mem = P.get('memory') or {}
        rows.append({
            'method': LABEL[m], 'n': len(df),
            'json_strict_%': 100 * df['json_strict'].mean() if m not in TYPED else np.nan,
            'schema_valid_%': 100 * df['schema_valid'].mean() if m not in TYPED else np.nan,
            'rule_consistent_%': 100 * df['rule_ok'].mean() if m not in TYPED else np.nan,
            'status_%': 100 * df['status'].mean(), 'department_%': 100 * df['department'].mean(),
            'urgency_%': 100 * df['urgency'].mean(), 'missing_info_%': 100 * df['missing_info'].mean(),
            'order_id_%': 100 * df['order_id'].mean() if 'order_id' in df else np.nan,
            'product_%': 100 * df['product'].mean() if 'product' in df else np.nan,
            'decision_%': 100 * df['decision'].mean(), 'exact_%': 100 * df['exact'].mean(),
            'latency_mean_s': lat.mean(), 'latency_p50_s': lat.median(), 'latency_p95_s': lat.quantile(0.95),
            'ttft_mean_s': df['ttft'].astype(float).mean(), 'tokens_in': df['n_in'].astype(float).mean(),
            'tokens_out': df['n_out'].astype(float).mean(), 'peak_gpu_mib': mem.get('peak_allocated_mib'),
            'usd_per_1k_requests': 1000 * usd, 'train_gpu_min': tsec / 60,
            'train_usd': tsec / 3600 * GPU_PRICE_PER_HOUR})
        for cat, g in df.groupby('category'):
            per_cat.append({'method': LABEL[m], 'category': cat, 'n': len(g), 'decision_%': 100 * g['decision'].mean(),
                            'exact_%': 100 * g['exact'].mean()})
    S = pd.DataFrame(rows)
    C = pd.DataFrame(per_cat)
    S.round(3).to_csv(out_path('hw2_summary.csv'), index=False)
    C.round(1).to_csv(out_path('hw2_by_category.csv'), index=False)
    return S, C


def plots(S, C):
    if S.empty:
        return
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    x = np.arange(len(S))
    for k, (col, lab) in enumerate((('schema_valid_%', 'schema valid'), ('decision_%', 'status + department'), ('exact_%', 'all fields'))):
        ax[0].bar(x + (k - 1) * 0.27, S[col].fillna(100 if col == 'schema_valid_%' else 0), 0.27, label=lab)
    ax[0].set_xticks(x); ax[0].set_xticklabels(S['method'], rotation=35, ha='right', fontsize=8)
    ax[0].set_ylabel('% of 52 benchmark cases'); ax[0].set_ylim(0, 105); ax[0].legend(fontsize=8)
    ax[0].set_title('Structure and accuracy (D: typed output, no parsing)')
    ax[1].bar(x, S['latency_mean_s'], color='tab:orange')
    ax[1].errorbar(x, S['latency_mean_s'], yerr=[np.zeros(len(S)), S['latency_p95_s'] - S['latency_mean_s']], fmt='none', ecolor='k', capsize=3)
    ax[1].set_xticks(x); ax[1].set_xticklabels(S['method'], rotation=35, ha='right', fontsize=8)
    ax[1].set_ylabel('seconds per request (mean, bar to p95)'); ax[1].set_title('Latency (batch 1)')
    ax[2].scatter(S['latency_mean_s'], S['decision_%'], s=60)
    for _, r in S.iterrows():
        ax[2].annotate(r['method'], (r['latency_mean_s'], r['decision_%']), fontsize=7, xytext=(4, 3), textcoords='offset points')
    ax[2].set_xscale('log'); ax[2].set_xlabel('mean latency (s, log)'); ax[2].set_ylabel('status + department correct (%)')
    ax[2].set_title('Accuracy vs. time')
    plt.tight_layout(); plt.savefig(out_path('hw2_comparison.png'), dpi=130); plt.show()
    if not C.empty:
        piv = C.pivot(index='method', columns='category', values='decision_%').reindex(S['method'])
        fig, ax = plt.subplots(figsize=(11, 0.5 * len(piv) + 1.5))
        im = ax.imshow(piv.values, cmap='RdYlGn', vmin=0, vmax=100, aspect='auto')
        ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns, rotation=30, ha='right')
        ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=8)
        ax.set_title('status + department correct (%) by benchmark category')
        plt.colorbar(im, ax=ax, fraction=0.03); plt.tight_layout(); plt.savefig(out_path('hw2_by_category.png'), dpi=130); plt.show()


def training_curves():
    fig, ax = plt.subplots(1, 4, figsize=(20, 3.8))
    for a, (d, title, key) in zip(ax, ((SFT_DIR, 'B: SFT loss', 'loss'), (DPO_DIR, 'B: DPO (+SFT term) loss', 'loss'),
                                       (S1_DIR_V2, 'D: System One v2, 3 epochs (sum of 7 CE)', 'loss'),
                                       (S1_DIR_LONG, 'D: System One v2, 10 epochs', 'loss'))):
        p = os.path.join(d, 'history.json')
        if not os.path.exists(p):
            a.set_title(title + ' (not trained)'); continue
        h = load_json(p)
        a.plot([s['step'] for s in h['steps']], [s[key] for s in h['steps']], lw=1, label='train (per update)')
        ev = [e for e in h['epochs_log'] if 'val_loss' in e]
        steps_per_epoch = math.ceil(h['n_train'] / h['grad_accum']) if 'n_train' in h else max(1, len(h['steps']) // max(1, h['epochs']))
        a.plot([e['epoch'] * steps_per_epoch for e in ev], [e['val_loss'] for e in ev], 'o-', label='validation')
        a.set_title(title); a.set_xlabel('optimizer step'); a.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_path('hw2_training_curves.png'), dpi=130); plt.show()


def run_all(gemini=True):
    bench, train, val = data('benchmark'), data('train'), data('val')
    t0 = time.time()
    stage_prompt_and_constrained(bench)
    stage_sft(bench, train, val)
    stage_dpo(bench, train, val)
    for version in ('v1', 'v2', 'v2_long'):
        stage_system_one(bench, train, val, version=version)
    if gemini:
        stage_gemini(bench)
    S, C = summary(bench)
    print(f'作業二 finished in {(time.time() - t0) / 60:.1f} min on {DEVICE}')
    return S, C
