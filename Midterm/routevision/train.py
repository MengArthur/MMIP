"""Training, prediction and metrics for all methods (cached checkpoints)."""

import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score,
                             roc_auc_score, accuracy_score)

from .dataset import PatchDataset, GRID, PATCH

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WORKERS = 2 if os.name != 'nt' else 0


# ---------------------------------------------------------------- metrics
def best_threshold(y, score):
    """Threshold maximising F1 of the hotspot class (chosen on validation data only)."""
    qs = np.unique(np.quantile(score, np.linspace(0.01, 0.99, 197)))
    f1s = [f1_score(y, score >= t, zero_division=0) for t in qs]
    return float(qs[int(np.argmax(f1s))])


def per_layout_f1(y, score, thr, patches=GRID * GRID):
    """Hotspot F1 of every layout separately (y, score: flat arrays ordered layout by layout)."""
    y = np.asarray(y).reshape(-1, patches).astype(bool)
    p = np.asarray(score).reshape(-1, patches) >= thr
    tp, fp, fn = (p & y).sum(1), (p & ~y).sum(1), (~p & y).sum(1)
    return 2 * tp / np.maximum(2 * tp + fp + fn, 1)


def metrics(y, score, thr):
    pred = score >= thr
    return {'ROC-AUC': roc_auc_score(y, score), 'PR-AUC': average_precision_score(y, score),
            'F1(hotspot)': f1_score(y, pred, zero_division=0), 'Macro-F1': f1_score(y, pred, average='macro', zero_division=0),
            'Precision': precision_score(y, pred, zero_division=0), 'Recall': recall_score(y, pred, zero_division=0),
            'Accuracy': accuracy_score(y, pred)}


# ---------------------------------------------------------------- torch models
def _loader(store, indices, bs, shuffle, augment=False):
    return torch.utils.data.DataLoader(PatchDataset(store, indices, augment), batch_size=bs, shuffle=shuffle,
                                       num_workers=WORKERS, pin_memory=DEVICE.type == 'cuda')


@torch.no_grad()
def predict(model, store, bs=256, amp=False):
    model.eval().to(DEVICE)
    out = []
    for x, side, _ in _loader(store, None, bs, False):
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=amp and DEVICE.type == 'cuda'):
            logit = model(x.to(DEVICE), side.to(DEVICE))
        out.append(torch.sigmoid(logit.float()).cpu())
    return torch.cat(out).numpy()


def train_torch(name, model, train_store, val_store, ckpt_dir, epochs=15, lr=1e-3, bs=128, wd=1e-4,
                amp=False, max_train=None, seed=42, augment=False, force=False, verbose=True):
    os.makedirs(ckpt_dir, exist_ok=True)
    pt, js = os.path.join(ckpt_dir, f'{name}.pt'), os.path.join(ckpt_dir, f'{name}.json')
    model.to(DEVICE)
    if os.path.exists(pt) and os.path.exists(js) and not force:
        model.load_state_dict(torch.load(pt, map_location=DEVICE))
        with open(js) as f:
            hist = json.load(f)
        print(f"[cache] {name}: best epoch {hist['best_epoch']}, val ROC-AUC {hist['best_val_auc']:.4f}")
        return hist
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(train_store))
    if max_train and len(idx) > max_train:
        idx = rng.choice(idx, max_train, replace=False)
    y = train_store.flat('y')[idx]
    pos_weight = torch.tensor((1 - y.mean()) / max(y.mean(), 1e-6), device=DEVICE)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * int(np.ceil(len(idx) / bs)),
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler('cuda') if amp and DEVICE.type == 'cuda' else None
    yv = val_store.flat('y')
    hist = {'train_loss': [], 'val_auc': []}
    best, best_state, best_ep, t0 = -1, None, 0, time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        tot, n, te = 0.0, 0, time.perf_counter()
        for x, side, t in _loader(train_store, idx, bs, True, augment):
            x, side, t = x.to(DEVICE, non_blocking=True), side.to(DEVICE), t.to(DEVICE)
            with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=scaler is not None):
                loss = F.binary_cross_entropy_with_logits(model(x, side).float(), t, pos_weight=pos_weight)
            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            sched.step()
            tot += loss.item() * len(t); n += len(t)
        auc = roc_auc_score(yv, predict(model, val_store, amp=amp))
        hist['train_loss'].append(round(tot / n, 5)); hist['val_auc'].append(round(float(auc), 5))
        if auc > best:
            best, best_ep = auc, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f'  [{name}] epoch {ep:2d}/{epochs} | loss {tot / n:.4f} | val ROC-AUC {auc:.4f} | {time.perf_counter() - te:.0f}s')
    model.load_state_dict(best_state)
    torch.save(best_state, pt)
    hist.update({'best_epoch': best_ep, 'best_val_auc': round(float(best), 5), 'epochs': epochs, 'lr': lr, 'augment': augment,
                 'train_windows': int(len(idx)), 'minutes': round((time.perf_counter() - t0) / 60, 2), 'device': str(DEVICE)})
    with open(js, 'w') as f:
        json.dump(hist, f, indent=1)
    return hist
