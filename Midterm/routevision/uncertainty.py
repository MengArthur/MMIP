"""
Uncertainty from ensemble disagreement, evaluated on the saved leave-one-design-out scores
(outputs/lodo/<design>/<setting>/{val,test}_scores.npz); no retraining.

    uncertainty of a patch = standard deviation of the ensemble members' hotspot probabilities
                             (Random Forest, Multimodal, ViT)
    baseline               = closeness of the ensemble score to its threshold, -|p - t|

Both are evaluated by the AUROC for "the ensemble's prediction is wrong" and by the share of the errors among
the 10 / 20 / 30 % most uncertain patches of the design (a label-free cut-off), plus the risk-coverage curve.
Also reported: the rule "uncertainty above the 80th percentile of the validation set" (fixed before looking at
the results); its flagged fraction on a new design shows how much more uncertain the model is there.

    python -m routevision.uncertainty      # -> outputs/uncertainty/uncertainty.csv, risk_coverage.json/.png
"""

import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

from .train import best_threshold

DESIGNS = ['Vortex-small', 'nvdla-small', 'openc910-1']
MEMBERS = ['OpenCV + RandomForest', 'Multimodal', 'ViT']
VAL_UNC_QUANTILE = 0.8
TOP_FRACTIONS = (0.1, 0.2, 0.3)


def disagreement(scores, members=MEMBERS):
    """Std of the members' hotspot probabilities, per patch (any shape)."""
    return np.std(np.stack([np.asarray(scores[m], np.float32) for m in members]), axis=0)


def captured(err, unc, frac):
    """Share of the errors among the `frac` most uncertain patches."""
    k = int(round(frac * len(unc)))
    top = np.zeros(len(unc), bool)
    top[np.argsort(-unc, kind='stable')[:k]] = True
    return float((err & top).sum() / max(err.sum(), 1))


def risk_coverage(err, unc, steps=19):
    """Error rate when only the `coverage` most certain patches are kept."""
    e = np.asarray(err)[np.argsort(unc, kind='stable')]
    return [{'coverage': round(float(c), 3), 'error': float(e[:max(1, int(round(c * len(e))))].mean())}
            for c in np.linspace(0.1, 1.0, steps)]


def load_fold(lodo_dir, design, setting):
    d = os.path.join(lodo_dir, design, setting)
    return dict(np.load(os.path.join(d, 'val_scores.npz'))), dict(np.load(os.path.join(d, 'test_scores.npz')))


def evaluate(lodo_dir='outputs/lodo', setting='aug', out_dir='outputs/uncertainty', designs=DESIGNS):
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    rows, curves = [], {}
    for design in designs:
        val, test = load_fold(lodo_dir, design, setting)
        y = test['y'].astype(bool)
        t = best_threshold(val['y'], val['Ensemble'])                   # same threshold as everywhere else
        err = (test['Ensemble'] >= t) != y
        measures = {'disagreement': disagreement(test), 'margin': -np.abs(test['Ensemble'] - t)}
        flag = measures['disagreement'] > float(np.quantile(disagreement(val), VAL_UNC_QUANTILE))
        for name, u in measures.items():
            rows.append({'test_design': design, 'measure': name, 'error_rate': float(err.mean()),
                         'AUROC_error': float(roc_auc_score(err, u)),
                         **{f'errors_in_top{int(f * 100)}%': captured(err, u, f) for f in TOP_FRACTIONS},
                         **({'flagged_by_val80': float(flag.mean()),
                             'errors_in_flagged': float((err & flag).sum() / max(err.sum(), 1)),
                             'error_rate_flagged': float(err[flag].mean()),
                             'error_rate_unflagged': float(err[~flag].mean())} if name == 'disagreement' else {})})
            curves.setdefault(design, {})[name] = risk_coverage(err, u)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, 'uncertainty.csv'), index=False)
    with open(os.path.join(out_dir, 'risk_coverage.json'), 'w') as f:
        json.dump(curves, f, indent=1)
    return df, curves


def plot(curves, out_dir='outputs/uncertainty'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Noto Sans CJK TC', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
    colors = {'Vortex-small': '#2a78d6', 'nvdla-small': '#eb6834', 'openc910-1': '#1baf7a'}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for d, cv in curves.items():
        for name, ls in (('disagreement', '-'), ('margin', ':')):
            rows = cv[name]
            ax.plot([r['coverage'] for r in rows], [r['error'] for r in rows], color=colors[d], lw=2, ls=ls,
                    label=f'{d} · {"模型間分歧" if name == "disagreement" else "離閾值遠近"}')
    ax.set_xlabel('保留的區塊比例（依不確定度由低到高）')
    ax.set_ylabel('Ensemble 錯誤率')
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'risk_coverage.png'), dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    df, curves = evaluate()
    plot(curves)
    print(df.groupby('measure', sort=False).mean(numeric_only=True).round(3).T)
