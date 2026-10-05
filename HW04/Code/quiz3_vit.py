"""
Quiz 3: fine-tune an ImageNet-pretrained Vision Transformer (ViT-B/16, basic) and
ResNet-50 (advanced) on Oxford-IIIT Pet, then compare their Macro-AUC.

Both models use the identical recipe (data, augmentation, optimiser, schedule,
epochs) so the comparison isolates the architecture.
"""

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

from .common import (DEVICE, NUM_WORKERS, plt, set_seed, count_params, fit_classifier, predict_proba,
                     topk_accuracy, plot_histories, save_fig_path)
from .data import HFImageDataset, IMAGENET_MEAN, IMAGENET_STD

EPOCHS = 8
BATCH = 32
LR = 1e-4
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 1
MODELS = [('ViT-B/16', 'vit_b_16'), ('ResNet-50', 'resnet50')]

TRAIN_TF = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0)), T.RandomHorizontalFlip(),
                      T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
EVAL_TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def build_model(arch, num_classes, pretrained=True):
    from torchvision import models
    if arch == 'vit_b_16':
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    elif arch == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(arch)
    return m


def _forward(model, batch):
    x, y = batch
    return model(x.to(DEVICE, non_blocking=True)), y.to(DEVICE, non_blocking=True)


def _loader(hf_ds, idx, tf, shuffle):
    return torch.utils.data.DataLoader(HFImageDataset(hf_ds, idx, tf), batch_size=BATCH, shuffle=shuffle,
                                       num_workers=NUM_WORKERS, pin_memory=DEVICE.type == 'cuda')


def macro_average_roc(y, probs, grid=np.logspace(-4, 0, 400)):
    """Average of the per-class (one-vs-rest) TPR curves on a common FPR grid."""
    Y = label_binarize(y, classes=np.arange(probs.shape[1]))
    tprs = []
    for c in range(probs.shape[1]):
        fpr, tpr, _ = roc_curve(Y[:, c], probs[:, c])
        tprs.append(np.interp(grid, fpr, tpr))
    return grid, np.mean(tprs, axis=0)


def _comparison_figure(results, y, class_names, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for name, r in results.items():
        fpr, tpr = macro_average_roc(y, r['probs'])
        axes[0].plot(fpr, tpr, label=f"{name} (Macro-AUC = {r['macro_auc']:.6f})")
    axes[0].set_xscale('log'); axes[0].set_xlim(1e-4, 1); axes[0].set_ylim(0, 1.01)
    axes[0].set_xlabel('False Positive Rate (log scale)'); axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('Macro-average ROC over 37 classes (test set)', fontweight='bold'); axes[0].legend(loc='lower right')
    (n1, r1), (n2, r2) = results.items()
    a1, a2 = r1['class_auc'], r2['class_auc']
    axes[1].scatter(a2, a1, s=18)
    lo = min(a1.min(), a2.min()) - 0.002
    axes[1].plot([lo, 1], [lo, 1], 'k--', lw=0.8)
    for c in np.argsort(np.minimum(a1, a2))[:6]:
        axes[1].annotate(class_names[c], (a2[c], a1[c]), fontsize=8, xytext=(4, -10), textcoords='offset points')
    axes[1].set_xlabel(f'{n2} per-class AUC'); axes[1].set_ylabel(f'{n1} per-class AUC')
    axes[1].set_title(f'Per-class AUC (points above the diagonal: {n1} better)', fontweight='bold')
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig


def _samples_figure(results, pets, idx, path):
    names = pets['class_names']
    fig, axes = plt.subplots(1, len(idx), figsize=(2.2 * len(idx), 3.6))
    for ax, i in zip(axes, idx):
        ax.imshow(pets['hf_test'][int(i)]['image'].convert('RGB')); ax.axis('off')
        true = pets['y_test'][i]
        lines = [f'true: {names[true]}']
        for name, r in results.items():
            p = r['probs'][i]
            lines.append(f'{name}: {names[p.argmax()]} {p.max():.2f}' + (' ✓' if p.argmax() == true else ' ✗'))
        ax.set_title('\n'.join(lines), fontsize=7)
    fig.suptitle('Quiz 3: test-set inference (✓ correct, ✗ wrong)', fontweight='bold')
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig


def run_quiz3(pets, ckpt_dir, output_dir, force=False, pretrained=True):
    print('=' * 70); print('QUIZ 3: VISION TRANSFORMER vs RESNET-50 (Oxford-IIIT Pet)'); print('=' * 70)
    names, y_test = pets['class_names'], pets['y_test']
    n_cls = len(names)
    train_ld = _loader(pets['hf_train'], pets['idx_train'], TRAIN_TF, True)
    val_ld = _loader(pets['hf_train'], pets['idx_val'], EVAL_TF, False)
    test_ld = _loader(pets['hf_test'], np.arange(len(y_test)), EVAL_TF, False)

    results, histories, rows = {}, {}, []
    for name, arch in MODELS:
        set_seed(42)
        model = build_model(arch, n_cls, pretrained)
        hist = fit_classifier(f'quiz3_{arch}', model, train_ld, val_ld, _forward, ckpt_dir, epochs=EPOCHS, lr=LR,
                              weight_decay=WEIGHT_DECAY, amp=True, warmup_epochs=WARMUP_EPOCHS, force=force)
        t0 = time.perf_counter()
        probs, y = predict_proba(model, test_ld, _forward, amp=True)
        secs = time.perf_counter() - t0
        Y = label_binarize(y, classes=np.arange(n_cls))
        class_auc = np.array([roc_auc_score(Y[:, c], probs[:, c]) for c in range(n_cls)])
        r = {'probs': probs, 'macro_auc': float(roc_auc_score(y, probs, multi_class='ovr', average='macro')),
             'class_auc': class_auc, 'top1': topk_accuracy(probs, y, 1), 'top5': topk_accuracy(probs, y, 5)}
        results[name], histories[name] = r, hist
        rows.append({'model': name, 'params': count_params(model), 'best_epoch': hist['best_epoch'],
                     'best_val_acc': hist['best_val_acc'], 'test_top1': round(r['top1'], 4),
                     'test_top5': round(r['top5'], 4), 'test_macro_auc': round(r['macro_auc'], 6),
                     'train_minutes': hist['train_minutes'],
                     'test_images_per_sec': round(len(y) / secs, 1)})
        del model
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()

    table = pd.DataFrame(rows)
    table.to_csv(save_fig_path(output_dir, 'quiz3_vit_vs_resnet.csv'), index=False)
    print(table.to_string(index=False))

    (n1, r1), (n2, r2) = results.items()
    per_class = pd.DataFrame({'class': names, 'n_test': np.bincount(y_test, minlength=n_cls)})
    for n, r in results.items():
        pred = r['probs'].argmax(1)
        per_class[f'{n} acc'] = [round(float((pred[y_test == c] == c).mean()), 4) for c in range(n_cls)]
        per_class[f'{n} AUC'] = r['class_auc'].round(6)
    per_class.to_csv(save_fig_path(output_dir, 'quiz3_per_class.csv'), index=False)

    preds = pd.DataFrame({'true': [names[c] for c in y_test]})
    confused = {}
    for n, r in results.items():
        pred = r['probs'].argmax(1)
        preds[f'{n} pred'] = [names[c] for c in pred]
        preds[f'{n} conf'] = r['probs'].max(1).round(4)
        wrong = pred != y_test
        pairs = pd.Series([f'{names[a]} -> {names[b]}' for a, b in zip(y_test[wrong], pred[wrong])])
        confused[n] = pairs.value_counts().head(5)
    preds.to_csv(save_fig_path(output_dir, 'quiz3_test_predictions.csv'), index_label='test_index')

    sample_idx = np.random.default_rng(0).choice(len(y_test), 8, replace=False)
    figs = {'curves': plot_histories(histories, save_fig_path(output_dir, 'quiz3_training_curves.png'),
                                     'Quiz 3: fine-tuning on Oxford-IIIT Pet'),
            'comparison': _comparison_figure(results, y_test, names, save_fig_path(output_dir, 'quiz3_macro_auc.png')),
            'samples': _samples_figure(results, pets, sample_idx, save_fig_path(output_dir, 'quiz3_sample_predictions.png'))}
    return {'table': table, 'per_class': per_class, 'confused': confused, 'figures': figs, 'histories': histories}
