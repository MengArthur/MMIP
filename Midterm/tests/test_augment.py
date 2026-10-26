"""Mirroring augmentation must equal taking the window from a mirrored layout (CPU, a few seconds).

    python -m pytest tests/test_augment.py -q
"""

import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from routevision import dataset as D


def make_store(flip=None, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.random((2, 4, D.SIZE, D.SIZE)).astype(np.float16)
    label = rng.random((2, D.SIZE, D.SIZE)).astype(np.float16)
    if flip == 'lr':
        X, label = X[..., ::-1], label[..., ::-1]
    elif flip == 'ud':
        X, label = X[..., ::-1, :], label[..., ::-1, :]
    data = {'d': {'X': np.ascontiguousarray(X), 'label': np.ascontiguousarray(label),
                  'egr': np.ascontiguousarray(label), 'params': rng.random((2, 6)).astype(np.float32)}}
    norm = D.Normalizer(np.ones(4, np.float32), np.zeros(6, np.float32), np.ones(6, np.float32))
    return D.LayoutStore(data, [('d', 0), ('d', 1)], norm)


def sample(store, k, flips):
    """Item k of an augmenting dataset with the two random flips forced to `flips`."""
    ds = D.PatchDataset(store, augment=True)
    orig = torch.rand
    torch.rand = lambda n: torch.tensor([0.0 if f else 1.0 for f in flips])
    try:
        return ds[k]
    finally:
        torch.rand = orig


def test_left_right_flip_matches_mirrored_layout():
    s, m = make_store(), make_store('lr')
    for layout, r, c in [(0, 0, 0), (0, 3, 5), (1, 7, 7), (1, 2, 0)]:
        x, side, y = sample(s, layout * 64 + r * 8 + c, (True, False))
        assert np.array_equal(x.numpy(), m.window(layout, r, D.GRID - 1 - c))
        assert np.allclose(side.numpy(), m.side(layout, r, D.GRID - 1 - c))
        assert float(y) == m.y[layout, r, D.GRID - 1 - c]


def test_up_down_flip_matches_mirrored_layout():
    s, m = make_store(), make_store('ud')
    for layout, r, c in [(0, 0, 0), (0, 3, 5), (1, 7, 7)]:
        x, side, y = sample(s, layout * 64 + r * 8 + c, (False, True))
        assert np.array_equal(x.numpy(), m.window(layout, D.GRID - 1 - r, c))
        assert np.allclose(side.numpy(), m.side(layout, D.GRID - 1 - r, c))
        assert float(y) == m.y[layout, D.GRID - 1 - r, c]


def test_no_augmentation_is_unchanged():
    s = make_store()
    x, side, y = D.PatchDataset(s)[64 + 3 * 8 + 5]
    assert np.array_equal(x.numpy(), s.window(1, 3, 5)) and np.allclose(side.numpy(), s.side(1, 3, 5))
