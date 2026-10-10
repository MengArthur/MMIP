"""
Quiz 1: image retrieval on CUB-200-2011 (unseen-class retrieval).

Model : ImageNet backbone (ResNet-50 or ViT-B/16) -> linear embedding (512-d) -> L2 normalised.
Loss  : Normalized Softmax (Zhai & Wu, 2019): cosine similarity to one learnable proxy per
        training class, divided by a temperature, then cross-entropy. Images of the same
        species are pulled together, different species pushed apart.
Eval  : every test image is a query; the gallery is all other test images (unseen species).
        Recall@K (K = 1, 2, 4, 8) and mAP. Pretrained features without training are the baseline.
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from .common import (DEVICE, NUM_WORKERS, IMAGENET_MEAN, IMAGENET_STD, plt, set_seed, count_params,
                     ckpt_paths, save_json, load_json, out_path, warmup_cosine, free_gpu)
from .data import ImageDataset

EMB_DIM = 512
EPOCHS = 12
BATCH = 64
LR_BACKBONE, LR_HEAD = 1e-4, 1e-3
TEMPERATURE = 0.05
KS = (1, 2, 4, 8)
BACKBONES = [('ResNet-50 (CNN)', 'resnet50'), ('ViT-B/16', 'vit_b_16')]

TRAIN_TF = T.Compose([T.RandomResizedCrop(224, scale=(0.5, 1.0)), T.RandomHorizontalFlip(),
                      T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
EVAL_TF = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class Embedder(nn.Module):
    def __init__(self, arch, emb_dim=EMB_DIM, pretrained=True):
        super().__init__()
        from torchvision import models
        if arch == 'resnet50':
            m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            feat_dim, m.fc = m.fc.in_features, nn.Identity()
        elif arch == 'vit_b_16':
            m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
            feat_dim, m.heads = m.heads.head.in_features, nn.Identity()
        else:
            raise ValueError(arch)
        self.backbone, self.feat_dim = m, feat_dim
        self.head = nn.Linear(feat_dim, emb_dim)

    def features(self, x):
        """Backbone features only (the untrained baseline)."""
        return F.normalize(self.backbone(x), dim=1)

    def forward(self, x):
        return F.normalize(self.head(self.backbone(x)), dim=1)


class NormSoftmax(nn.Module):
    def __init__(self, n_classes, emb_dim=EMB_DIM, temperature=TEMPERATURE):
        super().__init__()
        self.proxies = nn.Parameter(torch.randn(n_classes, emb_dim) * 0.01)
        self.t = temperature

    def forward(self, emb, y):
        return F.cross_entropy(emb @ F.normalize(self.proxies, dim=1).t() / self.t, y)


def _loader(cub, split, tf, shuffle):
    return torch.utils.data.DataLoader(ImageDataset(cub['hf'], cub['idx'][split], cub['labels'], tf),
                                       batch_size=BATCH, shuffle=shuffle, num_workers=NUM_WORKERS,
                                       pin_memory=DEVICE.type == 'cuda', drop_last=shuffle)


@torch.no_grad()
def embed(model, loader, raw_features=False):
    model.eval()
    embs, ys = [], []
    for x, y in loader:
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == 'cuda'):
            e = model.features(x.to(DEVICE)) if raw_features else model(x.to(DEVICE))
        embs.append(F.normalize(e.float(), dim=1).cpu()); ys.append(y)
    return torch.cat(embs).numpy(), torch.cat(ys).numpy()


def retrieval_metrics(emb, y, ks=KS):
    """Leave-one-out retrieval: each image queries all the others."""
    e = torch.from_numpy(emb).to(DEVICE)
    yy = torch.from_numpy(y).to(DEVICE)
    sim = e @ e.t()
    sim.fill_diagonal_(-2)                                       # never retrieve the query itself
    order = sim.argsort(dim=1, descending=True)
    rel = (yy[order] == yy[:, None]).float()                     # (N, N) relevance in ranked order
    rel = rel[:, :-1]                                            # last column is the query itself
    out = {f'R@{k}': float((rel[:, :k].sum(1) > 0).float().mean()) for k in ks}
    cum = rel.cumsum(1)
    prec = cum / torch.arange(1, rel.shape[1] + 1, device=DEVICE)
    ap = (prec * rel).sum(1) / rel.sum(1).clamp(min=1)
    out['mAP'] = float(ap.mean())
    return out, order[:, :10].cpu().numpy()


def train_embedder(name, arch, cub, ckpt_dir, force=False, epochs=EPOCHS):
    pt, js = ckpt_paths(ckpt_dir, name)
    set_seed(42)
    model = Embedder(arch).to(DEVICE)
    if not force and os.path.exists(pt) and os.path.exists(js):
        model.load_state_dict(torch.load(pt, map_location=DEVICE))
        hist = load_json(js)
        print(f"[cache] {name}: best epoch {hist['best_epoch']}, val R@1 {hist['best_val_r1']:.4f}")
        return model, hist
    crit = NormSoftmax(cub['n_train_classes']).to(DEVICE)
    opt = torch.optim.AdamW([{'params': model.backbone.parameters(), 'lr': LR_BACKBONE},
                             {'params': list(model.head.parameters()) + list(crit.parameters()), 'lr': LR_HEAD}],
                            weight_decay=1e-4)
    train_ld, val_ld = _loader(cub, 'train', TRAIN_TF, True), _loader(cub, 'val', EVAL_TF, False)
    sched = warmup_cosine(opt, epochs * len(train_ld), len(train_ld))
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None
    hist, best, best_state, t0 = {'train_loss': [], 'val_r1': []}, -1, None, time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train(); tot, n, te = 0.0, 0, time.perf_counter()
        for x, y in train_ld:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE)
            with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=scaler is not None):
                loss = crit(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            sched.step(); tot += loss.item() * len(y); n += len(y)
        r1 = retrieval_metrics(*embed(model, val_ld))[0]['R@1']
        hist['train_loss'].append(round(tot / n, 5)); hist['val_r1'].append(round(r1, 5))
        if r1 > best:
            best, hist['best_epoch'] = r1, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f'  [{name}] epoch {ep:2d}/{epochs} | loss {tot / n:.4f} | val R@1 {r1:.4f} | {time.perf_counter() - te:.0f}s')
    model.load_state_dict(best_state); torch.save(best_state, pt)
    hist.update({'best_val_r1': round(best, 5), 'epochs': epochs, 'train_minutes': round((time.perf_counter() - t0) / 60, 2),
                 'device': str(DEVICE)})
    save_json(hist, js)
    return model, hist


# ---------------------------------------------------------------- figures
def plot_examples(cub, test_idx, y, order, path, title, n_query=5, k=5, seed=0):
    rng = np.random.default_rng(seed)
    qs = rng.choice(len(y), n_query, replace=False)
    names = cub['class_names']
    fig, ax = plt.subplots(n_query, k + 1, figsize=(2.2 * (k + 1), 2.4 * n_query))
    for r, q in enumerate(qs):
        ax[r, 0].imshow(cub['hf'][int(test_idx[q])]['image'].convert('RGB'))
        ax[r, 0].set_title(f'QUERY\n{names[y[q]]}', fontsize=8, fontweight='bold')
        for j in range(k):
            g = order[q, j]
            ok = y[g] == y[q]
            ax[r, j + 1].imshow(cub['hf'][int(test_idx[g])]['image'].convert('RGB'))
            ax[r, j + 1].set_title(f'#{j + 1} {names[y[g]]}', fontsize=7, color='green' if ok else 'red')
            for s in ax[r, j + 1].spines.values():
                s.set_edgecolor('green' if ok else 'red'); s.set_linewidth(3)
        for a in ax[r]:
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f'{title}: top-{k} retrieved images (green = same species, red = different)', fontweight='bold')
    fig.tight_layout(); fig.savefig(path, dpi=130, bbox_inches='tight')
    return fig


def plot_recall(table, path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = 0.8 / len(table)
    for i, (_, r) in enumerate(table.iterrows()):
        vals = [r[f'R@{k}'] for k in KS]
        bars = ax.bar(np.arange(len(KS)) + (i - (len(table) - 1) / 2) * w, vals, w, label=r['model'])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f'{v:.2f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(range(len(KS)), [f'Recall@{k}' for k in KS]); ax.set_ylim(0, 1.05)
    ax.set_title('Retrieval on 100 unseen bird species (CUB-200-2011 test classes)', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig


def plot_history(hists, path):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for k, h in hists.items():
        ep = range(1, len(h['train_loss']) + 1)
        ax[0].plot(ep, h['train_loss'], label=k); ax[1].plot(ep, h['val_r1'], label=k)
        ax[1].axvline(h['best_epoch'], lw=0.8, alpha=0.5)
    ax[0].set(title='Training loss (Normalized Softmax)', xlabel='Epoch')
    ax[1].set(title='Validation Recall@1 (10 unseen species)', xlabel='Epoch')
    for a in ax:
        a.legend(fontsize=8); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig


def run_quiz1(cub, ckpt_dir, output_dir, force=False):
    print('=' * 70); print('QUIZ 1: IMAGE RETRIEVAL (CUB-200-2011, unseen species)'); print('=' * 70)
    for s in ('train', 'val', 'test'):
        cls = np.unique(cub['labels'][cub['idx'][s]])
        print(f'{s:5s}: {len(cub["idx"][s]):5d} images, {len(cls)} species ({cls.min() + 1}-{cls.max() + 1})')
    test_ld = _loader(cub, 'test', EVAL_TF, False)
    rows, hists, figs, orders = [], {}, {}, {}
    for label, arch in BACKBONES:
        # baseline: ImageNet features without retrieval training
        set_seed(42)
        base = Embedder(arch).to(DEVICE)
        t0 = time.perf_counter()
        emb, y = embed(base, test_ld, raw_features=True)
        m, _ = retrieval_metrics(emb, y)
        rows.append({'model': f'{label} pretrained (no training)', 'embedding_dim': emb.shape[1],
                     **{k: round(v, 4) for k, v in m.items()}, 'params': count_params(base.backbone),
                     'train_minutes': 0.0, 'images_per_sec': round(len(y) / (time.perf_counter() - t0), 1)})
        del base; free_gpu()
        model, hist = train_embedder(f'quiz1_{arch}', arch, cub, ckpt_dir, force=force)
        t0 = time.perf_counter()
        emb, y = embed(model, test_ld)
        m, order = retrieval_metrics(emb, y)
        rows.append({'model': f'{label} + retrieval training', 'embedding_dim': emb.shape[1],
                     **{k: round(v, 4) for k, v in m.items()}, 'params': count_params(model),
                     'train_minutes': hist['train_minutes'], 'images_per_sec': round(len(y) / (time.perf_counter() - t0), 1)})
        hists[label], orders[label] = hist, order
        figs[f'examples_{arch}'] = plot_examples(cub, cub['idx']['test'], y, order,
                                                 out_path(output_dir, f'quiz1_retrieval_examples_{arch}.png'), label)
        del model; free_gpu()
    table = pd.DataFrame(rows)
    table.to_csv(out_path(output_dir, 'quiz1_retrieval_results.csv'), index=False)
    print(table.to_string(index=False))
    figs['recall'] = plot_recall(table, out_path(output_dir, 'quiz1_recall_at_k.png'))
    figs['history'] = plot_history(hists, out_path(output_dir, 'quiz1_training_curves.png'))
    return {'table': table, 'figures': figs, 'histories': hists}
