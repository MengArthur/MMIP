"""
Patch-level hotspot dataset.

Each 256x256 layout is cut into an 8x8 grid of 32x32 target patches. A model sees
the 64x64 window centred on a patch (the patch plus 16 tiles of context on each
side) and predicts whether that patch is a congestion hotspot:

    hotspot  <=>  mean final global-routing utilisation inside the patch >= HOT_THRESHOLD

Splits are made by DESIGN (test design never seen in training); validation
layouts are a random subset of the training designs' layouts.
"""

import json
import os

import numpy as np
import torch

SIZE, PATCH, CTX = 256, 32, 16          # layout size, target patch, context on each side
GRID = SIZE // PATCH                     # 8 x 8 patches per layout
WIN = PATCH + 2 * CTX                    # 64 x 64 model input window
HOT_THRESHOLD = 0.6                      # set from the label statistics (see README)


def load_designs(processed_dir, designs):
    out = {}
    for d in designs:
        z = np.load(os.path.join(processed_dir, f'{d}.npz'))
        out[d] = {k: z[k] for k in z.files}
    return out


def patch_means(maps):
    """(N, 256, 256) -> (N, 8, 8) mean over each 32x32 patch."""
    n = maps.shape[0]
    return maps.astype(np.float32).reshape(n, GRID, PATCH, GRID, PATCH).mean(axis=(2, 4))


class Normalizer:
    """Per-channel scaling by the 99th percentile of non-zero training values; design
    parameters are z-scored. Fitted on training layouts only."""

    def __init__(self, scale=None, p_mean=None, p_std=None):
        self.scale, self.p_mean, self.p_std = scale, p_mean, p_std

    def fit(self, X, params):
        scale = []
        for c in range(X.shape[1]):
            v = X[:, c].astype(np.float32)
            v = v[v > 0]
            scale.append(float(np.percentile(v, 99)) if v.size else 1.0)
        self.scale = np.array(scale, dtype=np.float32)
        self.p_mean = params.mean(0).astype(np.float32)
        self.p_std = (params.std(0) + 1e-6).astype(np.float32)
        return self

    def x(self, X):
        return np.clip(X.astype(np.float32) / self.scale[None, :, None, None], 0, 3).astype(np.float32)

    def p(self, params):
        return ((params - self.p_mean) / self.p_std).astype(np.float32)

    def to_json(self, path):
        with open(path, 'w') as f:
            json.dump({'scale': self.scale.tolist(), 'p_mean': self.p_mean.tolist(), 'p_std': self.p_std.tolist()}, f)

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls(np.array(d['scale'], np.float32), np.array(d['p_mean'], np.float32), np.array(d['p_std'], np.float32))


def make_split(data, test_design, val_frac=0.15, seed=42):
    """Returns {'train': [(design, idx)], 'val': [...], 'test': [...]} at layout level."""
    rng = np.random.default_rng(seed)
    split = {'train': [], 'val': [], 'test': []}
    for d, a in data.items():
        idx = np.arange(len(a['X']))
        if d == test_design:
            split['test'] += [(d, i) for i in idx]
            continue
        rng.shuffle(idx)
        n_val = max(1, int(round(len(idx) * val_frac)))
        split['val'] += [(d, i) for i in idx[:n_val]]
        split['train'] += [(d, i) for i in idx[n_val:]]
    return split


class LayoutStore:
    """Normalised layouts of one split, held in memory, with patch labels and baselines."""

    def __init__(self, data, items, norm, threshold=HOT_THRESHOLD):
        self.items = items
        X = norm.x(np.stack([data[d]['X'][i] for d, i in items]))                           # (L, 4, 256, 256)
        self.Xpad = np.pad(X, ((0, 0), (0, 0), (CTX, CTX), (CTX, CTX)))
        self.X = self.Xpad[:, :, CTX:CTX + SIZE, CTX:CTX + SIZE]                             # view, no copy
        label = np.stack([data[d]['label'][i] for d, i in items]).astype(np.float32)
        self.label_map = label
        self.util = patch_means(label)                                                       # (L, 8, 8)
        self.y = (self.util >= threshold).astype(np.float32)
        self.rudy = patch_means(np.stack([data[d]['X'][i][1] for d, i in items]))           # raw RUDY baseline
        self.egr = patch_means(np.stack([data[d]['egr'][i] for d, i in items]))             # eGR baseline
        self.params = norm.p(np.stack([data[d]['params'][i] for d, i in items]))
        self.designs = np.array([d for d, _ in items])

    def __len__(self):
        return len(self.items) * GRID * GRID

    def unravel(self, k):
        layout, rc = divmod(k, GRID * GRID)
        return layout, rc // GRID, rc % GRID

    def window(self, layout, r, c):
        return self.Xpad[layout, :, r * PATCH:r * PATCH + WIN, c * PATCH:c * PATCH + WIN]

    def side(self, layout, r, c):
        """Design parameters + patch position (the second modality of the multimodal model)."""
        return np.concatenate([self.params[layout], [r / (GRID - 1), c / (GRID - 1)]]).astype(np.float32)

    def flat(self, key):
        return getattr(self, key).reshape(-1)


class PatchDataset(torch.utils.data.Dataset):
    """augment=True mirrors the whole layout at random (left-right and/or up-down). The window of patch
    (r, c) in a mirrored layout is the mirrored window of the original patch, now at column GRID-1-c
    (or row GRID-1-r), so the position features are mirrored with it. The label is unchanged: mirroring
    keeps horizontal wires horizontal, and a patch's mean utilisation does not depend on orientation."""

    def __init__(self, store, indices=None, augment=False):
        self.s = store
        self.idx = np.arange(len(store)) if indices is None else np.asarray(indices)
        self.augment = augment

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        layout, r, c = self.s.unravel(int(self.idx[i]))
        x, side = self.s.window(layout, r, c), self.s.side(layout, r, c)
        if self.augment:
            flip = torch.rand(2) < 0.5                     # torch RNG: seeded per DataLoader worker
            if flip[0]:
                x, side[-1] = x[:, :, ::-1], 1 - side[-1]    # left-right: column position mirrored
            if flip[1]:
                x, side[-2] = x[:, ::-1, :], 1 - side[-2]    # up-down: row position mirrored
        return torch.from_numpy(x.copy()), torch.from_numpy(side), torch.tensor(self.s.y[layout, r, c])
