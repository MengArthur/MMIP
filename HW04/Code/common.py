"""
Shared utilities: device, seeding, checkpoint caching, a generic classification
training loop (used by Quiz 2 text models and Quiz 3 vision models) and plotting.
"""

import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 2 if os.name != 'nt' else 0


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_params(model, trainable_only=False):
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def save_json(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def ckpt_paths(ckpt_dir, name):
    os.makedirs(ckpt_dir, exist_ok=True)
    return os.path.join(ckpt_dir, f'{name}.pt'), os.path.join(ckpt_dir, f'{name}.json')


def save_fig_path(output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, name)


def warmup_cosine(optimizer, total_steps, warmup_steps):
    """Linear warm-up followed by cosine decay, stepped once per mini-batch."""
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


# ----------------------------------------------------------------------------
# Generic classification training loop
# ----------------------------------------------------------------------------
def _run_epoch(model, loader, forward, optimizer=None, scheduler=None, scaler=None, clip=None):
    train = optimizer is not None
    model.train(train)
    use_amp = scaler is not None
    tot_loss, tot_correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
                logits, y = forward(model, batch)
                loss = F.cross_entropy(logits.float(), y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                if use_amp:
                    scaler.scale(loss).backward()
                    if clip:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            bs = y.size(0)
            tot_loss += loss.item() * bs
            tot_correct += (logits.argmax(1) == y).sum().item()
            n += bs
    return tot_loss / n, tot_correct / n


def fit_classifier(name, model, train_loader, val_loader, forward, ckpt_dir, epochs, lr,
                   weight_decay=0.0, clip=None, amp=False, warmup_epochs=0, force=False, verbose=True):
    """
    Trains with AdamW, keeps the weights of the epoch with the best validation
    accuracy, and caches them (<name>.pt) together with the history (<name>.json).
    A second call with the same name loads the cache instead of retraining.
    """
    pt, js = ckpt_paths(ckpt_dir, name)
    model.to(DEVICE)
    if os.path.exists(pt) and os.path.exists(js) and not force:
        model.load_state_dict(torch.load(pt, map_location=DEVICE))
        hist = load_json(js)
        print(f"[cache] {name}: loaded best-val checkpoint (epoch {hist['best_epoch']}, val acc {hist['best_val_acc']:.4f})")
        return hist

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    steps = epochs * len(train_loader)
    scheduler = warmup_cosine(optimizer, steps, warmup_epochs * len(train_loader)) if warmup_epochs else None
    scaler = torch.amp.GradScaler('cuda') if amp and DEVICE.type == 'cuda' else None

    hist = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_acc, best_state, best_epoch = -1.0, None, 0
    t_start = time.perf_counter()
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = _run_epoch(model, train_loader, forward, optimizer, scheduler, scaler, clip)
        va_loss, va_acc = _run_epoch(model, val_loader, forward)
        for k, v in zip(hist, (tr_loss, tr_acc, va_loss, va_acc)):
            hist[k].append(round(v, 5))
        if va_acc > best_acc:
            best_acc, best_epoch = va_acc, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"  [{name}] epoch {ep:2d}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f} | {time.perf_counter() - t0:.0f}s")

    model.load_state_dict(best_state)
    torch.save(best_state, pt)
    hist.update({'name': name, 'epochs': epochs, 'best_epoch': best_epoch, 'best_val_acc': round(best_acc, 5),
                 'train_minutes': round((time.perf_counter() - t_start) / 60, 2), 'device': str(DEVICE),
                 'lr': lr, 'weight_decay': weight_decay})
    save_json(hist, js)
    return hist


@torch.no_grad()
def predict_proba(model, loader, forward, amp=False):
    model.eval()
    probs, ys = [], []
    for batch in loader:
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=amp and DEVICE.type == 'cuda'):
            logits, y = forward(model, batch)
        probs.append(torch.softmax(logits.float(), 1).cpu())
        ys.append(y.cpu())
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def topk_accuracy(probs, y, k=1):
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float((topk == np.asarray(y)[:, None]).any(1).mean())


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def plot_histories(histories, path, title, right_key='acc', right_label='Accuracy'):
    """histories: {label: history}. Left = loss, right = `right_key` (train solid, val dashed)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, (label, h) in enumerate(histories.items()):
        c = colors[i % len(colors)]
        ep = range(1, len(h['train_loss']) + 1)
        axes[0].plot(ep, h['train_loss'], '-', color=c, label=f'{label} (train)')
        axes[0].plot(ep, h['val_loss'], '--', color=c, label=f'{label} (val)')
        if f'train_{right_key}' in h:
            axes[1].plot(ep, h[f'train_{right_key}'], '-', color=c, label=f'{label} (train)')
        axes[1].plot(ep, h[f'val_{right_key}'], '--', color=c, label=f'{label} (val)')
        axes[1].axvline(h['best_epoch'], color=c, lw=0.8, alpha=0.5)
    axes[0].set_title('Cross-Entropy Loss', fontweight='bold')
    axes[1].set_title(f'{right_label} (thin vertical line = best epoch kept)', fontweight='bold')
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(title + '  (solid = train, dashed = validation)', fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    return fig
