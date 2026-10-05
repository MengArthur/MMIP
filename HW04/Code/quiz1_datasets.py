"""
Quiz 1: dataset preparation for Sentiment Analysis (IMDB), Image Classification
(Oxford-IIIT Pet) and Image Captioning (Flickr8k).
"""

import textwrap

import numpy as np
import pandas as pd

from .common import plt, save_fig_path
from .data import tokenize, tokenize_caption, IMDB_ID, PETS_ID, FLICKR_ID


def _imdb_figure(imdb, max_len, path):
    lens = np.array([len(tokenize(t)) for t in imdb['train'][0]])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    splits = ['train', 'val', 'test']
    neg = [int((imdb[s][1] == 0).sum()) for s in splits]
    pos = [int((imdb[s][1] == 1).sum()) for s in splits]
    x = np.arange(3)
    axes[0].bar(x - 0.2, neg, 0.4, label='negative', color='#d62728')
    axes[0].bar(x + 0.2, pos, 0.4, label='positive', color='#2ca02c')
    for xi, (a, b) in enumerate(zip(neg, pos)):
        axes[0].text(xi - 0.2, a, f'{a:,}', ha='center', va='bottom', fontsize=8)
        axes[0].text(xi + 0.2, b, f'{b:,}', ha='center', va='bottom', fontsize=8)
    axes[0].set_xticks(x, [f'{s}\n({len(imdb[s][1]):,})' for s in splits])
    axes[0].set_title('IMDB: class balance per split', fontweight='bold'); axes[0].legend()
    axes[1].hist(np.clip(lens, 0, 1000), bins=60, color='#1f77b4')
    axes[1].axvline(max_len, color='k', ls='--', label=f'max_len = {max_len} ({(lens > max_len).mean():.0%} of reviews are longer)')
    axes[1].set_title('IMDB: review length in tokens (train, clipped at 1000)', fontweight='bold')
    axes[1].set_xlabel('tokens'); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig, lens


def _pets_figure(pets, path, n=12, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pets['hf_test']), n, replace=False)
    fig = plt.figure(figsize=(16, 7.5))
    gs = fig.add_gridspec(3, 7, width_ratios=[1, 1, 1, 1, 0.15, 1.6, 1.6])
    for k, i in enumerate(idx):
        ax = fig.add_subplot(gs[k // 4, k % 4])
        ex = pets['hf_test'][int(i)]
        ax.imshow(ex['image'].convert('RGB')); ax.axis('off')
        ax.set_title(pets['class_names'][ex['label']], fontsize=9)
    ax = fig.add_subplot(gs[:, 5:])
    c = len(pets['class_names'])
    counts = {s: np.bincount(pets[f'y_{s}'], minlength=c) for s in ('train', 'val', 'test')}
    left = np.zeros(c)
    for s, col in zip(counts, ['#1f77b4', '#ff7f0e', '#2ca02c']):
        ax.barh(np.arange(c), counts[s], left=left, color=col, label=f'{s} ({counts[s].sum():,})')
        left += counts[s]
    ax.set_yticks(np.arange(c), pets['class_names'], fontsize=7); ax.invert_yaxis()
    ax.set_title('Oxford-IIIT Pet: images per class (37 breeds)', fontweight='bold'); ax.legend(fontsize=8)
    fig.suptitle('Quiz 1: Oxford-IIIT Pet samples (test split) and split sizes', fontweight='bold')
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig


def _flickr_figure(flickr, path, n=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(flickr['captions']['test']), n, replace=False)
    lens = np.array([len(tokenize_caption(c)) for caps in flickr['captions']['train'] for c in caps])
    fig = plt.figure(figsize=(16, 6.5))
    gs = fig.add_gridspec(2, n + 1, height_ratios=[1.3, 1])
    for k, i in enumerate(idx):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(flickr['hf']['test'][int(i)]['image'].convert('RGB')); ax.axis('off')
        ax_t = fig.add_subplot(gs[1, k]); ax_t.axis('off')
        txt = '\n'.join(textwrap.fill(f'{j + 1}. {c}', 52) for j, c in enumerate(flickr['captions']['test'][i]))
        ax_t.text(0, 1, txt, va='top', fontsize=8)
    ax = fig.add_subplot(gs[:, n])
    ax.hist(lens, bins=range(0, 40), color='#9467bd')
    ax.set_title('Caption length in words (train, 30,000 captions)', fontweight='bold'); ax.set_xlabel('words')
    fig.suptitle('Quiz 1: Flickr8k test images with their 5 human-written reference captions', fontweight='bold')
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig, lens


def run_quiz1(imdb, pets, flickr, output_dir, max_len):
    print('=' * 70); print('QUIZ 1: DATASETS'); print('=' * 70)
    figs = {}
    figs['imdb'], imdb_lens = _imdb_figure(imdb, max_len, save_fig_path(output_dir, 'quiz1_imdb_overview.png'))
    figs['pets'] = _pets_figure(pets, save_fig_path(output_dir, 'quiz1_pets_overview.png'))
    figs['flickr'], cap_lens = _flickr_figure(flickr, save_fig_path(output_dir, 'quiz1_flickr8k_overview.png'))

    caps = flickr['captions']
    rows = [
        {'task': 'Sentiment Analysis (Quiz 2)', 'dataset': 'IMDB Large Movie Review', 'hf_id': IMDB_ID,
         'classes / targets': 'negative / positive', 'train': len(imdb['train'][1]), 'val': len(imdb['val'][1]),
         'test': len(imdb['test'][1]), 'notes': f'median review length {int(np.median(imdb_lens))} tokens'},
        {'task': 'Image Classification (Quiz 3)', 'dataset': 'Oxford-IIIT Pet', 'hf_id': PETS_ID,
         'classes / targets': f"{len(pets['class_names'])} cat & dog breeds", 'train': len(pets['y_train']),
         'val': len(pets['y_val']), 'test': len(pets['y_test']), 'notes': 'fine-grained, roughly 100 train images per class'},
        {'task': 'Image Captioning (Quiz 4)', 'dataset': 'Flickr8k', 'hf_id': FLICKR_ID,
         'classes / targets': '5 reference captions per image', 'train': len(caps['train']),
         'val': len(caps['validation']), 'test': len(caps['test']),
         'notes': f'{5 * len(caps["train"]):,} training captions, median {int(np.median(cap_lens))} words'},
    ]
    table = pd.DataFrame(rows)
    table.to_csv(save_fig_path(output_dir, 'quiz1_dataset_summary.csv'), index=False)
    print(table.to_string(index=False))
    return {'table': table, 'figures': figs}
