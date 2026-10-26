"""Hand-crafted (OpenCV) window features for the Random Forest baseline."""

import cv2
import numpy as np

from .dataset import CTX, PATCH, GRID


def window_features(win, side):
    """win: (4, 64, 64) normalised window; side: (8,) params + position -> 1-D feature vector."""
    centre = win[:, CTX:CTX + PATCH, CTX:CTX + PATCH]
    feats = []
    for c in range(win.shape[0]):
        cc = centre[c]
        feats += [cc.mean(), cc.std(), cc.max(), win[c].mean(), (cc > 0.5).mean()]
    for c in (0, 1):                                    # edges of cell density and routing demand
        gx = cv2.Sobel(win[c], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(win[c], cv2.CV_32F, 0, 1, ksize=3)
        feats.append(np.sqrt(gx ** 2 + gy ** 2)[CTX:CTX + PATCH, CTX:CTX + PATCH].mean())
    return np.concatenate([np.array(feats, np.float32), side])


FEATURE_NAMES = ([f'{ch}_{s}' for ch in ('density', 'rudy', 'rudy_pin', 'macro')
                  for s in ('mean', 'std', 'max', 'ctx_mean', 'frac>0.5')]
                 + ['density_grad', 'rudy_grad'] + ['freq', 'mp', 'fpu', 'fpa', 'p', 'fi', 'row', 'col'])


def store_features(store):
    """All windows of a LayoutStore -> (n_windows, n_features)."""
    out = []
    for layout in range(len(store.items)):
        for r in range(GRID):
            for c in range(GRID):
                out.append(window_features(store.window(layout, r, c), store.side(layout, r, c)))
    return np.stack(out)
