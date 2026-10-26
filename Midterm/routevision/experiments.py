"""
End-to-end experiment: split by design, train / load every method, evaluate on the
held-out design, and save tables, figures and the artefacts the Streamlit app needs.
"""

import json
import os
import shutil
import sys
import time

import joblib
import numpy as np
import pandas as pd
import matplotlib
if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, precision_recall_curve

from . import dataset as D
from .features import store_features, FEATURE_NAMES
from .gradcam import layout_cam
from .models import build
from .train import DEVICE, train_torch, predict, best_threshold, metrics, per_layout_f1

# epochs / learning rates for the Colab T4 run
CONFIG = {
    'MLP': dict(epochs=15, lr=1e-3, bs=256),
    'CNN': dict(epochs=15, lr=1e-3, bs=256),
    'Multimodal': dict(epochs=15, lr=1e-3, bs=256),
    'ViT': dict(epochs=3, lr=5e-5, bs=64, amp=True, max_train=30000, wd=0.05),
}
RF_PARAMS = dict(n_estimators=300, min_samples_leaf=5, class_weight='balanced', n_jobs=-1, random_state=42)
ENSEMBLE = ('OpenCV + RandomForest', 'Multimodal', 'ViT')     # probabilities averaged
LEARNED = ('OpenCV + RandomForest', 'MLP', 'CNN', 'Multimodal', 'ViT')


def prepare(processed_dir, designs, test_design, threshold=D.HOT_THRESHOLD, ckpt_dir='checkpoints'):
    data = D.load_designs(processed_dir, designs)
    split, norm, stores = make_stores(data, test_design, threshold, ckpt_dir)
    return data, split, norm, stores


def make_stores(data, test_design, threshold=D.HOT_THRESHOLD, ckpt_dir='checkpoints'):
    """Split by design, fit the normaliser on the training layouts only, build train / val / test stores."""
    split = D.make_split(data, test_design)
    train_X = np.stack([data[d]['X'][i] for d, i in split['train']])
    train_p = np.stack([data[d]['params'][i] for d, i in split['train']])
    norm = D.Normalizer().fit(train_X, train_p)
    os.makedirs(ckpt_dir, exist_ok=True)
    norm.to_json(os.path.join(ckpt_dir, 'norm.json'))
    stores = {k: D.LayoutStore(data, v, norm, threshold) for k, v in split.items()}
    for k, s in stores.items():
        print(f'{k:5s}: {len(s.items):4d} layouts, {len(s):6d} windows, hotspot rate {s.y.mean():.3f}, '
              f'designs {sorted(set(map(str, s.designs)))}')
    return split, norm, stores


def run(stores, ckpt_dir='checkpoints', out_dir='outputs', methods=('MLP', 'CNN', 'Multimodal', 'ViT'),
        config=None, force=False, model_kwargs=None, augment=False, rf_params=None,
        ensemble=ENSEMBLE):
    """augment: random mirroring while training the deep models. ensemble: methods whose probabilities
    are averaged into an extra 'Ensemble' method (threshold chosen on validation like every other method)."""
    cfg = {**CONFIG, **(config or {})}
    model_kwargs = model_kwargs or {}
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    tr, va, te = stores['train'], stores['val'], stores['test']
    yv, yt = va.flat('y'), te.flat('y')
    scores, vscores, thresholds, timing, models = {}, {}, {}, {}, {}

    # 1) EDA estimates used directly as scores (no learning)
    for name, key in [('RUDY (EDA estimate)', 'rudy'), ('eGR (EDA estimate)', 'egr')]:
        vscores[name] = va.flat(key)
        thresholds[name] = best_threshold(yv, vscores[name])
        scores[name] = te.flat(key)

    # 2) OpenCV features + Random Forest
    rf_path = os.path.join(ckpt_dir, 'RandomForest.joblib')
    t0 = time.perf_counter()
    if os.path.exists(rf_path) and not force:
        rf = joblib.load(rf_path); print('[cache] RandomForest')
    else:
        rf = RandomForestClassifier(**{**RF_PARAMS, **(rf_params or {})})
        rf.fit(store_features(tr), tr.flat('y'))
        joblib.dump(rf, rf_path)
    name = 'OpenCV + RandomForest'
    vscores[name] = rf.predict_proba(store_features(va))[:, 1]
    thresholds[name] = best_threshold(yv, vscores[name])
    scores[name] = rf.predict_proba(store_features(te))[:, 1]
    timing[name] = time.perf_counter() - t0
    pd.Series(rf.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False).to_csv(
        os.path.join(out_dir, 'rf_feature_importance.csv'), header=['importance'])

    # 3) deep models
    hist = {}
    for m in methods:
        c = dict(cfg[m])
        model = build(m, **model_kwargs.get(m, {}))
        hist[m] = train_torch(m, model, tr, va, ckpt_dir, force=force, augment=augment, **c)
        amp = c.get('amp', False)
        vscores[m] = predict(model, va, amp=amp)
        thresholds[m] = best_threshold(yv, vscores[m])
        scores[m] = predict(model, te, amp=amp)
        models[m] = model

    # 4) ensemble of the strongest methods (no extra training)
    if ensemble and all(m in scores for m in ensemble):
        vscores['Ensemble'] = np.mean([vscores[m] for m in ensemble], 0)
        thresholds['Ensemble'] = best_threshold(yv, vscores['Ensemble'])
        scores['Ensemble'] = np.mean([scores[m] for m in ensemble], 0)
        with open(os.path.join(ckpt_dir, 'ensemble.json'), 'w') as f:
            json.dump({'members': list(ensemble)}, f)

    rows = [{'method': k, **{m: round(v, 4) for m, v in metrics(yt, s, thresholds[k]).items()},
             'threshold (from val)': round(thresholds[k], 4)} for k, s in scores.items()]
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, 'test_results.csv'), index=False)
    with open(os.path.join(ckpt_dir, 'thresholds.json'), 'w') as f:
        json.dump(thresholds, f, indent=1)
    np.savez_compressed(os.path.join(out_dir, 'test_scores.npz'), y=yt, **{k: v for k, v in scores.items()})
    np.savez_compressed(os.path.join(out_dir, 'val_scores.npz'), y=yv, **{k: v for k, v in vscores.items()})
    return {'table': table, 'scores': scores, 'thresholds': thresholds, 'models': models, 'history': hist}


# ---------------------------------------------------------------- leave-one-design-out
def cross_design(processed_dir, designs, settings=(('no_aug', False), ('aug', True)), ckpt_root='checkpoints_lodo',
                 out_root='outputs/lodo', methods=('MLP', 'CNN', 'Multimodal', 'ViT'), config=None, rf_params=None,
                 ensemble=ENSEMBLE, threshold=D.HOT_THRESHOLD):
    """Every design is the test design once (trained on the other designs), with and without augmentation.
    Checkpoints are cached per fold / setting, so an interrupted run resumes where it stopped."""
    data = D.load_designs(processed_dir, designs)
    rows = []
    for test in designs:
        print(f'\n=== test design: {test} ===')
        for setting, aug in settings:
            ck, out = os.path.join(ckpt_root, test, setting), os.path.join(out_root, test, setting)
            _, _, stores = make_stores(data, test, threshold, ck)
            print(f'--- {setting} ---')
            res = run(stores, ck, out, methods, config=config, augment=aug, rf_params=rf_params, ensemble=ensemble)
            yt = stores['test'].flat('y')
            ref = per_layout_f1(yt, res['scores']['eGR (EDA estimate)'], res['thresholds']['eGR (EDA estimate)'])
            for _, r in res['table'].iterrows():
                m = r['method']
                f = per_layout_f1(yt, res['scores'][m], res['thresholds'][m])
                rows.append({'test_design': test, 'setting': setting, 'hotspot_rate': round(float(yt.mean()), 4),
                             **r.to_dict(), 'mean_layout_F1': round(float(f.mean()), 4),
                             'layouts_beating_eGR': int((f > ref).sum()), 'layouts': len(f)})
    df = pd.DataFrame(rows)
    os.makedirs(out_root, exist_ok=True)
    df.to_csv(os.path.join(out_root, 'all_folds.csv'), index=False)
    summary = summarize_cross_design(df)
    summary.to_csv(os.path.join(out_root, 'summary.csv'), index=False)
    return df, summary


def summarize_cross_design(df):
    """Mean and standard deviation over the test designs for every method and setting."""
    keys = ['ROC-AUC', 'PR-AUC', 'F1(hotspot)', 'Macro-F1', 'mean_layout_F1']
    g = df.groupby(['setting', 'method'], sort=False)
    out = g[keys].agg(['mean', 'std']).round(4)
    out.columns = [f'{k} {s}' for k, s in out.columns]
    wins = g['layouts_beating_eGR'].sum().astype(str) + ' / ' + g['layouts'].sum().astype(str)
    out['layouts_beating_eGR'] = wins.where(out.index.get_level_values('method') != 'eGR (EDA estimate)', '(reference)')
    return out.reset_index()


def choose_setting(df, methods=LEARNED):
    """Setting with the higher mean ROC-AUC of the learned methods, averaged over all test designs."""
    s = df[df['method'].isin(methods)].groupby('setting')['ROC-AUC'].mean()
    return s.idxmax(), s.round(4).to_dict()


def plot_cross_design(df, path, metric='ROC-AUC'):
    """Mean over test designs (bars, one per setting) with every design as a marker."""
    methods = list(dict.fromkeys(df['method']))
    settings = list(dict.fromkeys(df['setting']))
    designs = list(dict.fromkeys(df['test_design']))
    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(methods)), 4.5))
    w = 0.8 / len(settings)
    markers = ['o', 's', '^', 'D']
    for i, st in enumerate(settings):
        sub = df[df['setting'] == st].set_index(['method', 'test_design'])[metric]
        x = np.arange(len(methods)) + (i - (len(settings) - 1) / 2) * w
        ax.bar(x, [sub[m].mean() for m in methods], w * 0.9, label=f'{st} (mean)', alpha=0.75)
        for j, d in enumerate(designs):
            ax.scatter(x, [sub[(m, d)] for m in methods], s=14, color='k', marker=markers[j % 4], zorder=3,
                       label=d if i == 0 else None)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels([m.replace(' (EDA estimate)', '').replace('OpenCV + ', '') for m in methods], fontsize=9)
    ax.set_ylim(max(0, df[metric].min() - 0.05), 1.0)
    ax.set(ylabel=metric, title=f'Leave-one-design-out: {metric} on the held-out design')
    ax.legend(fontsize=8, ncol=2, loc='lower right', framealpha=0.95); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig


def export_final(ckpt_root, out_root, test_design, setting, ckpt_dir='checkpoints', out_dir='outputs',
                 backup='checkpoints_v1'):
    """Copy one fold's artefacts into checkpoints/ and outputs/ (what the web app loads).
    The previous checkpoints/ is kept once as checkpoints_v1/."""
    src_ck, src_out = os.path.join(ckpt_root, test_design, setting), os.path.join(out_root, test_design, setting)
    if os.path.exists(ckpt_dir) and not os.path.exists(backup):
        shutil.copytree(ckpt_dir, backup)
    os.makedirs(ckpt_dir, exist_ok=True); os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(src_ck):
        shutil.copy2(os.path.join(src_ck, f), os.path.join(ckpt_dir, f))
    for f in ('test_results.csv', 'test_scores.npz', 'val_scores.npz', 'rf_feature_importance.csv'):
        shutil.copy2(os.path.join(src_out, f), os.path.join(out_dir, f))
    print(f'exported {test_design} / {setting} -> {ckpt_dir}/, {out_dir}/')


# ---------------------------------------------------------------- figures
def plot_curves(yt, scores, path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for k, s in scores.items():
        fpr, tpr, _ = roc_curve(yt, s)
        p, r, _ = precision_recall_curve(yt, s)
        ls = '--' if 'EDA' in k else '-'
        ax[0].plot(fpr, tpr, ls, label=k); ax[1].plot(r, p, ls, label=k)
    ax[0].plot([0, 1], [0, 1], 'k:', lw=0.8)
    ax[0].set(xlabel='False Positive Rate', ylabel='True Positive Rate', title='ROC (held-out design)')
    ax[1].axhline(yt.mean(), color='k', ls=':', lw=0.8)
    ax[1].set(xlabel='Recall', ylabel='Precision', title='Precision-Recall (dotted = hotspot rate)')
    for a in ax:
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig


def plot_history(hist, path):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for k, h in hist.items():
        ep = range(1, len(h['train_loss']) + 1)
        ax[0].plot(ep, h['train_loss'], label=k); ax[1].plot(ep, h['val_auc'], label=k)
    ax[0].set(title='Training loss', xlabel='Epoch'); ax[1].set(title='Validation ROC-AUC', xlabel='Epoch')
    for a in ax:
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig


def plot_layouts(store, scores, thresholds, layouts, path, methods=None):
    """Inputs, ground truth and each method's hotspot map for a few held-out layouts."""
    methods = methods or list(scores)
    n = len(store.items)
    cols = 3 + len(methods)
    fig, ax = plt.subplots(len(layouts), cols, figsize=(2.3 * cols, 2.5 * len(layouts)), squeeze=False)
    for r, L in enumerate(layouts):
        ax[r, 0].imshow(store.X[L, 1], cmap='inferno'); ax[r, 0].set_title('RUDY input', fontsize=8)
        ax[r, 1].imshow(store.label_map[L], cmap='inferno', vmin=0, vmax=1); ax[r, 1].set_title('true utilisation', fontsize=8)
        ax[r, 2].imshow(store.y[L], cmap='Reds', vmin=0, vmax=1); ax[r, 2].set_title('true hotspots', fontsize=8)
        for j, m in enumerate(methods):
            s = scores[m].reshape(n, D.GRID, D.GRID)[L]
            ax[r, 3 + j].imshow(s >= thresholds[m], cmap='Reds', vmin=0, vmax=1)
            ax[r, 3 + j].set_title(m.replace(' (EDA estimate)', '').replace('OpenCV + ', ''), fontsize=8)
        for a in ax[r]:
            a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches='tight')
    return fig


def plot_gradcam(model, store, layouts, path, title='CNN'):
    fig, ax = plt.subplots(len(layouts), 3, figsize=(9, 3 * len(layouts)), squeeze=False)
    for r, L in enumerate(layouts):
        cam, prob = layout_cam(model, store, L)
        ax[r, 0].imshow(store.X[L, 1], cmap='inferno'); ax[r, 0].set_title('RUDY input', fontsize=9)
        ax[r, 1].imshow(np.kron(prob, np.ones((D.PATCH, D.PATCH))), cmap='Reds', vmin=0, vmax=1)
        ax[r, 1].set_title(f'{title}: hotspot probability', fontsize=9)
        ax[r, 2].imshow(store.X[L, 0], cmap='gray'); ax[r, 2].imshow(cam, cmap='jet', alpha=0.45)
        ax[r, 2].set_title('Grad-CAM over cell density', fontsize=9)
        for a in ax[r]:
            a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches='tight')
    return fig


def export_demo(data, split, out_path, n=6, seed=0):
    """Small file of held-out layouts for the Streamlit demo (raw values, as produced by prepare.py)."""
    rng = np.random.default_rng(seed)
    pick = [split['test'][i] for i in sorted(rng.choice(len(split['test']), min(n, len(split['test'])), replace=False))]
    np.savez_compressed(out_path,
                        X=np.stack([data[d]['X'][i] for d, i in pick]), label=np.stack([data[d]['label'][i] for d, i in pick]),
                        egr=np.stack([data[d]['egr'][i] for d, i in pick]), params=np.stack([data[d]['params'][i] for d, i in pick]),
                        samples=np.array([str(data[d]['samples'][i]) for d, i in pick]), designs=np.array([d for d, _ in pick]))
    return out_path
