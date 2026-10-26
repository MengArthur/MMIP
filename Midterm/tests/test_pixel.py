"""Pixel-level U-Net (routevision/pixel.py, models.UNet): metrics, augmentation, model shapes.

    python -m pytest tests/test_pixel.py -q
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from routevision import pixel as P
from routevision.models import UNet


def test_metrics_on_known_maps():
    rng = np.random.default_rng(0)
    t = rng.random((3, 64, 64)).astype(np.float32)
    assert np.allclose(P.pearson(t, t), 1) and np.allclose(P.ssim(t, t), 1) and np.allclose(P.nrmse(t, t), 0)
    assert np.allclose(P.pearson(2 * t + 1, t), 1)                  # Pearson ignores a linear rescaling
    assert np.allclose(P.pearson(1 - t, t), -1)
    assert (P.ssim(t + 0.3, t) < 0.99).all()                        # SSIM does not
    noisy = np.clip(t + rng.normal(0, 0.2, t.shape), 0, 1).astype(np.float32)
    assert (P.pearson(noisy, t) < 0.95).all() and (P.ssim(noisy, t) < P.ssim(t, t)).all()


def test_linear_calibration_recovers_the_line():
    x = np.random.default_rng(1).random((4, 32, 32))
    a, b = P.fit_linear(x, 0.7 * x + 0.1)
    assert a == pytest.approx(0.7, abs=1e-6) and b == pytest.approx(0.1, abs=1e-6)


class _Store:
    def __init__(self):
        self.items = [0, 1]
        self.X = np.arange(2 * 4 * 8 * 8, dtype=np.float32).reshape(2, 4, 8, 8)
        self.label_map = self.X[:, 1].copy()                       # label = channel 1, so flips must match
        self.params = np.ones((2, 6), np.float32)


def test_augmentation_flips_input_and_label_together():
    ds = P.MapDataset(_Store(), augment=True)
    torch.manual_seed(0)
    for _ in range(20):
        x, p, y = ds[0]
        assert torch.equal(x[1], y) and p.shape == (6,)


@pytest.mark.parametrize('use_params', [False, True])
def test_unet_shapes_and_film(use_params):
    torch.manual_seed(0)
    m = UNet(use_params=use_params, pretrained=False).eval()
    x, p = torch.rand(2, 4, 256, 256), torch.randn(2, 6)
    with torch.no_grad():
        y = m(x, p)
    assert y.shape == (2, 256, 256) and float(y.min()) >= 0 and float(y.max()) <= 1
    if use_params:                                                  # FiLM starts as the identity...
        with torch.no_grad():
            assert torch.allclose(m(x, p), m(x, torch.zeros_like(p)), atol=1e-6)
            m.film[-1].bias.fill_(0.5)                              # ...and changes the output once trained
            assert float((m(x, p) - y).abs().max()) > 1e-6         # (small: an untrained decoder is nearly constant)
