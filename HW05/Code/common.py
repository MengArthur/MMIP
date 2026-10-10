"""Shared utilities: device, seeding, JSON / checkpoint helpers, LR schedule."""

import json
import math
import os
import random
import sys

import numpy as np
import torch
import matplotlib
if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: F401  (re-exported for the quiz modules)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 2 if os.name != 'nt' else 0
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


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


def out_path(output_dir, name):
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, name)


def warmup_cosine(optimizer, total_steps, warmup_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def free_gpu(*_):
    """Release cached GPU memory (callers drop their own references first)."""
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
