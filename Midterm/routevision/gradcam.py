"""Grad-CAM for the CNN-based models, per window and stitched over a whole layout."""

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import GRID, PATCH, CTX, WIN, SIZE


def window_cam(model, x, side):
    """x: (4, 64, 64), side: (8,) -> (64, 64) CAM in [0, 1] and the hotspot probability."""
    acts, grads = {}, {}
    layer = model.gradcam_layer()
    h1 = layer.register_forward_hook(lambda m, i, o: acts.__setitem__('a', o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__('g', go[0]))
    model.eval()
    dev = next(model.parameters()).device
    xb = torch.as_tensor(x)[None].to(dev).requires_grad_(True)
    logit = model(xb, torch.as_tensor(side)[None].to(dev))
    model.zero_grad()
    logit.sum().backward()
    h1.remove(); h2.remove()
    a, g = acts['a'][0].detach(), grads['g'][0].detach()
    cam = F.relu((g.mean(dim=(1, 2))[:, None, None] * a).sum(0))
    cam = F.interpolate(cam[None, None], size=WIN, mode='bilinear', align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.cpu().numpy(), float(torch.sigmoid(logit).item())


def layout_cam(model, store, layout):
    """Average the window CAMs of all 64 patches back onto the 256x256 layout."""
    acc = np.zeros((SIZE + 2 * CTX, SIZE + 2 * CTX), np.float32)
    cnt = np.zeros_like(acc)
    prob = np.zeros((GRID, GRID), np.float32)
    for r in range(GRID):
        for c in range(GRID):
            cam, p = window_cam(model, store.window(layout, r, c), store.side(layout, r, c))
            acc[r * PATCH:r * PATCH + WIN, c * PATCH:c * PATCH + WIN] += cam * p   # weight by predicted hotspot prob
            cnt[r * PATCH:r * PATCH + WIN, c * PATCH:c * PATCH + WIN] += 1
            prob[r, c] = p
    m = (acc / np.maximum(cnt, 1))[CTX:CTX + SIZE, CTX:CTX + SIZE]
    return m / (m.max() + 1e-8), prob


def unet_cam(model, x, params, threshold, layer='d3'):
    """Seg-Grad-CAM (Vinogradova et al. 2020) for the pixel-level U-Net.
    x: (4, 256, 256), params: (6,). Target: the predicted utilisation summed over the patches the model calls
    hotspots (patch mean >= threshold; the whole map if there are none). Returns a (256, 256) CAM in [0, 1].
    Default layer d3 (decoder, 32x32): the bottleneck l4 is only 8x8 (as coarse as the patch grid) and l3 lit up
    the layout borders; d3 is sharper and still sees the encoder features through the skip connections."""
    acts, grads = {}, {}
    mod = getattr(model, layer)
    h1 = mod.register_forward_hook(lambda m, i, o: acts.__setitem__('a', o))
    h2 = mod.register_full_backward_hook(lambda m, gi, go: grads.__setitem__('g', go[0]))
    model.eval()
    dev = next(model.parameters()).device
    xb = torch.as_tensor(np.ascontiguousarray(x))[None].to(dev).requires_grad_(True)
    pred = model(xb, torch.as_tensor(params)[None].to(dev))[0]
    with torch.no_grad():
        hot = (F.avg_pool2d(pred[None, None], PATCH)[0, 0] >= threshold).float()
        mask = F.interpolate(hot[None, None], size=SIZE, mode='nearest')[0, 0]
        if mask.sum() == 0:
            mask = torch.ones_like(pred)
    model.zero_grad()
    (pred * mask).sum().backward()
    h1.remove(); h2.remove()
    a, g = acts['a'][0].detach(), grads['g'][0].detach()
    cam = F.relu((g.mean(dim=(1, 2))[:, None, None] * a).sum(0))
    cam = F.interpolate(cam[None, None], size=SIZE, mode='bilinear', align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.cpu().numpy()
