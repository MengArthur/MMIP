"""
MMIP HW02: Shared Utilities and Evaluation Metrics
Provides functions for confusion matrix, classification metrics,
ROC curve computation, plotting utilities, and runtime benchmarking.
"""

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score
)


def compute_classification_metrics(y_true, y_prob, threshold=0.5):
    """
    Compute Accuracy, Precision, Recall, F1-Score, and Confusion Matrix
    based on a given classification probability threshold.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        'threshold': float(threshold),
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'confusion_matrix': cm,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'y_pred': y_pred
    }


def compute_roc_auc(y_true, y_prob):
    """
    Compute ROC Curve coordinates (FPR, TPR, thresholds) and AUC score.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    return {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'auc': float(auc)
    }


def benchmark_function(func, *args, N=10, warmup=2, **kwargs):
    """
    Execute func N times with warmup to measure runtime in milliseconds.
    Returns: (mean_ms, std_ms, result)
    """
    for _ in range(warmup):
        res = func(*args, **kwargs)

    timings = []
    for _ in range(N):
        t0 = time.perf_counter()
        res = func(*args, **kwargs)
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(timings))
    std_ms = float(np.std(timings))
    return mean_ms, std_ms, res


def render_confusion_matrix_axis(ax, cm, title, class_names=None, subtitle=None):
    """
    Draw a clean, annotated confusion matrix on a matplotlib axis.
    Shows raw counts and percentages with high contrast.
    """
    if class_names is None:
        class_names = ['Negative (0)', 'Positive (1)']

    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    full_title = title if not subtitle else f"{title}\n{subtitle}"
    ax.set_title(full_title, fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=10, fontweight='bold')

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, fontsize=9)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=9)

    total = np.sum(cm)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = (val / total) * 100.0 if total > 0 else 0
            color = "white" if val > thresh else "black"
            cell_text = f"{val:,}\n({pct:.1f}%)"
            ax.text(j, i, cell_text, ha="center", va="center", color=color, fontsize=11, fontweight='bold')
