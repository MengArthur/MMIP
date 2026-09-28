"""
Evaluation and plotting helpers: Top-1/Top-5 accuracy, test-set prediction
summaries, per-class ROC curves with Macro-AUC, and training-curve plots.
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

from .data import CLASS_NAMES
from .train import predict_proba, topk_accuracy


def evaluate(model, x, y):
    probs = predict_proba(model, x)
    y = np.asarray(y)
    return {
        'probs': probs,
        'pred': probs.argmax(1),
        'top1': topk_accuracy(probs, y, 1),
        'top5': topk_accuracy(probs, y, 5),
        'macro_auc': float(roc_auc_score(y, probs, multi_class='ovr', average='macro')),
    }


def prediction_table(result, y, csv_path=None):
    """Per-sample predictions (true, predicted, confidence, top-5) -> DataFrame / CSV."""
    probs, y = result['probs'], np.asarray(y)
    top5 = np.argsort(-probs, axis=1)[:, :5]
    df = pd.DataFrame({
        'index': np.arange(len(y)),
        'true_label': [CLASS_NAMES[i] for i in y],
        'pred_label': [CLASS_NAMES[i] for i in result['pred']],
        'confidence': probs.max(1).round(4),
        'correct': result['pred'] == y,
        'top5': [', '.join(CLASS_NAMES[j] for j in row) for row in top5],
    })
    if csv_path:
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    return df


def per_class_table(result, y):
    y = np.asarray(y)
    rows = []
    for c, name in enumerate(CLASS_NAMES):
        mask = y == c
        top5 = np.argsort(-result['probs'][mask], axis=1)[:, :5]
        rows.append({
            'class': name,
            'n_test': int(mask.sum()),
            'top1_acc': float((result['pred'][mask] == c).mean()),
            'top5_acc': float(np.mean([c in r for r in top5])),
            'auc': float(roc_auc_score((y == c).astype(int), result['probs'][:, c])),
        })
    return pd.DataFrame(rows)


def plot_confusion_matrices(results, y, titles, path):
    fig, axes = plt.subplots(1, len(results), figsize=(8 * len(results), 7))
    axes = np.atleast_1d(axes)
    for ax, res, title in zip(axes, results, titles):
        cm = confusion_matrix(y, res['pred'])
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks(range(10), CLASS_NAMES, rotation=45, ha='right')
        ax.set_yticks(range(10), CLASS_NAMES)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
        ax.set_title(f"{title}\nTop-1 {res['top1']:.4f} | Top-5 {res['top5']:.4f}", fontweight='bold')
        for i in range(10):
            for j in range(10):
                ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=8,
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle('Test-Set Confusion Matrices (10,000 images)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig


def plot_sample_predictions(x_test, y, results, titles, path, n=12, seed=0):
    """Same random test images for every model; green = correct, red = wrong."""
    idx = np.random.default_rng(seed).choice(len(y), n, replace=False)
    fig, axes = plt.subplots(len(results), n, figsize=(1.6 * n, 2.1 * len(results)))
    axes = np.atleast_2d(axes)
    for r, (res, title) in enumerate(zip(results, titles)):
        for c, i in enumerate(idx):
            ax = axes[r, c]
            ax.imshow(x_test[i].permute(1, 2, 0).numpy())
            ok = res['pred'][i] == y[i]
            ax.set_title(f"{CLASS_NAMES[res['pred'][i]]}\n{res['probs'][i].max():.2f}",
                         fontsize=8, color='green' if ok else 'red')
            ax.axis('off')
        axes[r, 0].text(-0.3, 0.5, title, transform=axes[r, 0].transAxes, ha='right', va='center',
                        fontsize=9, fontweight='bold')
    fig.suptitle('Sample Test Predictions (title = predicted class & confidence; green = correct, red = wrong)\n'
                 'True labels: ' + ', '.join(CLASS_NAMES[y[i]] for i in idx), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig


def plot_roc_curves(results, y, titles, path):
    y = np.asarray(y)
    fig, axes = plt.subplots(1, len(results), figsize=(7.5 * len(results), 6.5))
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap('tab10')
    for ax, res, title in zip(axes, results, titles):
        for c, name in enumerate(CLASS_NAMES):
            fpr, tpr, _ = roc_curve((y == c).astype(int), res['probs'][:, c])
            auc = roc_auc_score((y == c).astype(int), res['probs'][:, c])
            ax.plot(fpr, tpr, color=cmap(c), lw=1.5, label=f'{name} (AUC={auc:.3f})')
        ax.plot([0, 1], [0, 1], '--', color='gray', lw=1)
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title(f"{title}\nMacro-AUC = {res['macro_auc']:.4f}", fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle('Per-Class ROC Curves (One-vs-Rest) on Test Set', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig


def plot_histories(histories, labels, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    cmap = plt.get_cmap('tab10')
    for k, (h, lab) in enumerate(zip(histories, labels)):
        ep = np.arange(1, len(h['train_loss']) + 1)
        axes[0].plot(ep, h['train_loss'], color=cmap(k), lw=1.8, label=f'{lab} (train)')
        axes[0].plot(ep, h['val_loss'], color=cmap(k), lw=1.8, ls='--', label=f'{lab} (val)')
        axes[1].plot(ep, h['train_acc'], color=cmap(k), lw=1.8, label=f'{lab} (train)')
        axes[1].plot(ep, h['val_acc'], color=cmap(k), lw=1.8, ls='--', label=f'{lab} (val)')
    axes[0].set_title('Cross-Entropy Loss', fontweight='bold'); axes[0].set_xlabel('Epoch')
    axes[1].set_title('Top-1 Accuracy', fontweight='bold'); axes[1].set_xlabel('Epoch')
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig


def experiment_row(name, history, test_res):
    return {
        'experiment': name,
        'epochs': len(history['train_loss']),
        'best_epoch': history['best_epoch'],
        'final_train_acc': round(history['train_acc'][-1], 4),
        'best_val_acc': round(history['best_val_acc'], 4),
        'test_top1': round(test_res['top1'], 4),
        'test_top5': round(test_res['top5'], 4),
        'test_macro_auc': round(test_res['macro_auc'], 4),
        'train_minutes': round(sum(history['epoch_seconds']) / 60, 1),
        'trainable_params': history['trainable_params'],
    }


def save_fig_path(output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, name)
