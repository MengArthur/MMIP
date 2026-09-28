"""
Training / evaluation loop shared by every experiment, with on-disk caching:
each experiment saves its best-validation checkpoint (.pt) and training history
(.json) under Checkpoints/, so HW03_main.py trains once and the notebook reuses
the exact same trained models. Delete Checkpoints/ (or pass force=True) to retrain.
"""

import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from .data import normalize, augment_batch
from .models import PlainCNN, ResNet18Classifier, count_params


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(cfg):
    if cfg['arch'] == 'plain':
        return PlainCNN(width=cfg.get('width', 32), dropout=cfg.get('dropout', 0.3))
    return ResNet18Classifier(pretrained=cfg.get('pretrained', True),
                              freeze_backbone=cfg.get('freeze_backbone', False),
                              input_size=cfg.get('input_size', 64))


@torch.no_grad()
def predict_proba(model, x_uint8, batch_size=500):
    model.eval()
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(x_uint8), batch_size):
        out.append(torch.softmax(model(normalize(x_uint8[i:i + batch_size]).to(device)), dim=1).cpu())
    return torch.cat(out).numpy()


def topk_accuracy(probs, y, k=1):
    y = np.asarray(y)
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([y[i] in topk[i] for i in range(len(y))]))


def train_model(cfg, data, ckpt_dir, force=False, verbose=True):
    """
    cfg keys: name, arch ('plain'|'resnet18'), epochs, lr, batch_size, augment,
              plus model-specific keys (dropout, width, pretrained, freeze_backbone).
    Returns (model loaded with best-val weights, history dict).
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{cfg['name']}.pt")
    hist_path = os.path.join(ckpt_dir, f"{cfg['name']}.json")

    set_seed(cfg.get('seed', 42))
    model = build_model(cfg).to(DEVICE)

    if not force and os.path.exists(ckpt_path) and os.path.exists(hist_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        with open(hist_path, encoding='utf-8') as f:
            history = json.load(f)
        if verbose:
            print(f"[cache] {cfg['name']}: loaded best-val checkpoint (epoch {history['best_epoch']}, "
                  f"val acc {history['best_val_acc']:.4f})")
        return model, history

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=cfg['lr'], weight_decay=cfg.get('weight_decay', 0.0))
    criterion = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(cfg.get('seed', 42))

    x_tr, y_tr = data['x_train'], data['y_train']
    x_val, y_val = data['x_val'], data['y_val']
    bs = cfg['batch_size']
    history = {'config': cfg, 'device': str(DEVICE), 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
               'epoch_seconds': [], 'total_params': count_params(model),
               'trainable_params': count_params(model, trainable_only=True)}
    best_acc, best_state = -1.0, None

    for epoch in range(1, cfg['epochs'] + 1):
        t0 = time.perf_counter()
        model.train()
        perm = torch.randperm(len(x_tr), generator=gen)
        run_loss, run_correct = 0.0, 0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            xb = x_tr[idx]
            if cfg.get('augment', False):
                xb = augment_batch(xb, gen)
            xb, yb = normalize(xb).to(DEVICE), y_tr[idx].to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * len(idx)
            run_correct += (logits.argmax(1) == yb).sum().item()

        val_probs = predict_proba(model, x_val)
        val_loss = float(nn.functional.nll_loss(torch.log(torch.from_numpy(val_probs) + 1e-12), y_val))
        val_acc = topk_accuracy(val_probs, y_val.numpy(), 1)

        history['train_loss'].append(run_loss / len(x_tr))
        history['train_acc'].append(run_correct / len(x_tr))
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch_seconds'].append(time.perf_counter() - t0)

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history['best_epoch'] = epoch
            history['best_val_acc'] = val_acc

        if verbose:
            print(f"  [{cfg['name']}] epoch {epoch:2d}/{cfg['epochs']} | "
                  f"train loss {history['train_loss'][-1]:.4f} acc {history['train_acc'][-1]:.4f} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.4f} | {history['epoch_seconds'][-1]:.1f}s")

    model.load_state_dict(best_state)
    torch.save(best_state, ckpt_path)
    with open(hist_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    return model, history
