"""
Quiz 1: dataset preparation -- source/content description, train/val/test split
and the image-classification problem definition.
"""

import sys

import numpy as np
import matplotlib
if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .data import CLASS_NAMES
from .evaluation import save_fig_path


def run_quiz1(data, output_dir, per_class=8, seed=0):
    counts = {s: np.bincount(data[f'y_{s}'].numpy(), minlength=10) for s in ('train', 'val', 'test')}

    print("=" * 70)
    print("QUIZ 1: DATASET PREPARATION (CIFAR-10)")
    print("=" * 70)
    print("Source : CIFAR-10 (Krizhevsky, 2009), official release by uoft-cs")
    print("Content: 60,000 RGB images, 32x32 px, 10 mutually exclusive classes")
    print(f"Split  : train {len(data['y_train'])} | val {len(data['y_val'])} (stratified from official train set, seed 42) "
          f"| test {len(data['y_test'])} (official test set)")
    print(f"{'class':12s} {'train':>6s} {'val':>5s} {'test':>5s}")
    for c, name in enumerate(CLASS_NAMES):
        print(f"{name:12s} {counts['train'][c]:6d} {counts['val'][c]:5d} {counts['test'][c]:5d}")

    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(10, per_class + 6)
    y_tr = data['y_train'].numpy()
    for c, name in enumerate(CLASS_NAMES):
        idx = rng.choice(np.where(y_tr == c)[0], per_class, replace=False)
        for j, i in enumerate(idx):
            ax = fig.add_subplot(gs[c, j])
            ax.imshow(data['x_train'][i].permute(1, 2, 0).numpy())
            ax.axis('off')
            if j == 0:
                ax.text(-0.2, 0.5, name, transform=ax.transAxes, ha='right', va='center', fontsize=9)
    ax = fig.add_subplot(gs[:, per_class + 1:])
    pos = np.arange(10)
    for k, (s, col) in enumerate([('train', '#1f77b4'), ('val', '#ff7f0e'), ('test', '#2ca02c')]):
        ax.barh(pos + (k - 1) * 0.27, counts[s], height=0.27, color=col, label=f'{s} ({counts[s].sum():,})')
    ax.set_yticks(pos, CLASS_NAMES); ax.invert_yaxis()
    ax.set_xlabel('Images per class'); ax.legend(); ax.grid(axis='x', alpha=0.3)
    ax.set_title('Class distribution (balanced)', fontweight='bold')
    fig.suptitle('Quiz 1: CIFAR-10 Samples (8 random training images per class) & Split Sizes',
                 fontsize=13, fontweight='bold')
    path = save_fig_path(output_dir, 'quiz1_dataset_overview.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"[Saved] {path}")
    return {'counts': {k: v.tolist() for k, v in counts.items()}, 'figure': fig, 'plot_path': path}
