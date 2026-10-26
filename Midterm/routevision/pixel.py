"""
Pixel-level prediction of the final global-routing utilisation map with a U-Net (ImageNet ResNet-18 encoder),
leave-one-design-out like every other method. Still a discriminative (supervised) model: input the placement
feature maps, output the utilisation of every pixel, labelled by the real routing result.

Two variants (an ablation of the second modality):
    'U-Net'               image only
    'U-Net (multimodal)'  + the 6 design parameters through FiLM at the bottleneck
Rule fixed before training: the web app uses the variant with the higher mean pixel Pearson over the 3 designs.

Evaluation on the held-out design
    pixel level  Pearson r, SSIM, NRMSE (RMSE / range of the true map) per layout, averaged; against the EDA maps
                 eGR (raw, same unit), eGR and RUDY linearly calibrated on the training layouts
    patch level  the predicted map averaged over each 32x32 patch is the hotspot score, the threshold is chosen on
                 validation -> the same table as the other methods (ROC-AUC, F1, ..., layouts beating eGR)

    from routevision import pixel as P
    P.cross_design('data/processed', DESIGNS)          # Colab T4: about 20 minutes
    P.export('Vortex-small')                           # -> checkpoints/UNet.*, outputs/test_*.npz, test_results.csv
"""

import json
import os
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import dataset as D
from .models import UNet
from .train import DEVICE, WORKERS, best_threshold, metrics, per_layout_f1

VARIANTS = {'U-Net': False, 'U-Net (multimodal)': True}
CONFIG = dict(epochs=60, lr=3e-4, bs=8, wd=1e-4, amp=True, seed=42, augment=True)
EGR = 'eGR (EDA estimate)'
BASELINE_MAPS = ['eGR (EDA estimate)', 'eGR (linear calibrated)', 'RUDY (linear calibrated)']


# ---------------------------------------------------------------- data
class MapDataset(torch.utils.data.Dataset):
    """Whole layouts: normalised 4-channel input, normalised design parameters, utilisation map.
    augment mirrors input and label together (left-right / up-down); the parameters do not change."""

    def __init__(self, store, augment=False):
        self.s, self.augment = store, augment

    def __len__(self):
        return len(self.s.items)

    def __getitem__(self, i):
        x, y = self.s.X[i], self.s.label_map[i]
        if self.augment:
            flip = torch.rand(2) < 0.5
            if flip[0]:
                x, y = x[:, :, ::-1], y[:, ::-1]
            if flip[1]:
                x, y = x[:, ::-1, :], y[::-1, :]
        return (torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(self.s.params[i]),
                torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)))


# ---------------------------------------------------------------- training
def train_map(name, use_params, train_store, val_store, ckpt_dir, epochs=60, lr=3e-4, bs=8, wd=1e-4, amp=True,
              seed=42, augment=True, force=False, pretrained=True, verbose=True):
    """MSE regression of the utilisation map; the epoch with the lowest validation MSE is kept.
    Cached as <ckpt_dir>/<file>.pt / .json, like the other models."""
    os.makedirs(ckpt_dir, exist_ok=True)
    stem = file_stem(name)
    pt, js = os.path.join(ckpt_dir, f'{stem}.pt'), os.path.join(ckpt_dir, f'{stem}.json')
    torch.manual_seed(seed)
    model = UNet(use_params=use_params, pretrained=pretrained).to(DEVICE)
    if os.path.exists(pt) and os.path.exists(js) and not force:
        model.load_state_dict(torch.load(pt, map_location=DEVICE))
        with open(js) as f:
            hist = json.load(f)
        print(f"[cache] {name}: best epoch {hist['best_epoch']}, val MSE {hist['best_val_mse']:.5f}")
        return model, hist
    loader = torch.utils.data.DataLoader(MapDataset(train_store, augment), batch_size=bs, shuffle=True,
                                         num_workers=WORKERS, drop_last=False)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * len(loader), pct_start=0.1)
    use_amp = amp and DEVICE.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    hist = {'train_mse': [], 'val_mse': [], 'val_pearson': []}
    best, best_state, best_ep, t0 = float('inf'), None, 0, time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for x, p, y in loader:
            x, p, y = x.to(DEVICE), p.to(DEVICE), y.to(DEVICE)
            with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=use_amp):
                pred = model(x, p)
            loss = F.mse_loss(pred.float(), y)
            opt.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()
            sched.step()
            tot += loss.item() * len(y); n += len(y)
        vp = predict_maps(model, val_store, amp=amp)
        v_mse = float(np.mean((vp - val_store.label_map) ** 2))
        v_r = float(np.mean(pearson(vp, val_store.label_map)))
        hist['train_mse'].append(round(tot / n, 6)); hist['val_mse'].append(round(v_mse, 6)); hist['val_pearson'].append(round(v_r, 4))
        if v_mse < best:
            best, best_ep = v_mse, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f'  [{name}] epoch {ep:2d}/{epochs} | train MSE {tot / n:.5f} | val MSE {v_mse:.5f} | val Pearson {v_r:.4f}')
    model.load_state_dict(best_state)
    torch.save(best_state, pt)
    hist.update({'name': name, 'use_params': use_params, 'best_epoch': best_ep, 'best_val_mse': round(best, 6),
                 'epochs': epochs, 'lr': lr, 'augment': augment, 'train_layouts': len(train_store.items),
                 'params_M': round(sum(q.numel() for q in model.parameters()) / 1e6, 2),
                 'minutes': round((time.perf_counter() - t0) / 60, 2), 'device': str(DEVICE)})
    with open(js, 'w') as f:
        json.dump(hist, f, indent=1)
    return model, hist


def file_stem(name):
    return {'U-Net': 'UNet', 'U-Net (multimodal)': 'UNet_multimodal'}[name]


@torch.no_grad()
def predict_maps(model, store, bs=8, amp=True):
    model.eval()
    out = []
    for i in range(0, len(store.items), bs):
        x = torch.from_numpy(np.ascontiguousarray(store.X[i:i + bs])).to(DEVICE)
        p = torch.from_numpy(store.params[i:i + bs]).to(DEVICE)
        with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=amp and DEVICE.type == 'cuda'):
            out.append(model(x, p).float().cpu())
    return torch.cat(out).numpy()


def patch_scores(maps):
    return D.patch_means(maps).reshape(-1)


# ---------------------------------------------------------------- pixel metrics (per layout)
def pearson(pred, true):
    p = pred.reshape(len(pred), -1).astype(np.float64)
    t = true.reshape(len(true), -1).astype(np.float64)
    p, t = p - p.mean(1, keepdims=True), t - t.mean(1, keepdims=True)
    return (p * t).sum(1) / np.sqrt((p ** 2).sum(1) * (t ** 2).sum(1) + 1e-12)


def nrmse(pred, true):
    t = true.reshape(len(true), -1)
    rmse = np.sqrt(((pred.reshape(len(pred), -1) - t) ** 2).mean(1))
    return rmse / (t.max(1) - t.min(1) + 1e-8)


def ssim(pred, true, data_range=1.0, win=11, sigma=1.5):
    """Standard SSIM (Wang et al. 2004) with an 11x11 Gaussian window, per layout."""
    g = torch.exp(-(torch.arange(win, dtype=torch.float64) - win // 2) ** 2 / (2 * sigma ** 2))
    g = (g / g.sum())
    k = (g[:, None] * g[None, :])[None, None]
    a = torch.from_numpy(np.asarray(pred, np.float64))[:, None]
    b = torch.from_numpy(np.asarray(true, np.float64))[:, None]
    f = lambda z: F.conv2d(z, k)
    mu_a, mu_b = f(a), f(b)
    va, vb, cov = f(a * a) - mu_a ** 2, f(b * b) - mu_b ** 2, f(a * b) - mu_a * mu_b
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))
    return s.mean(dim=(1, 2, 3)).numpy()


def pixel_metrics(pred, true):
    return {'Pearson': float(np.mean(pearson(pred, true))), 'SSIM': float(np.mean(ssim(pred, true))),
            'NRMSE': float(np.mean(nrmse(pred, true)))}


# ---------------------------------------------------------------- EDA maps as pixel baselines
def raw_maps(data, items, key):
    if key == 'rudy':
        return np.stack([data[d]['X'][i][1] for d, i in items]).astype(np.float32)
    return np.stack([data[d][key][i] for d, i in items]).astype(np.float32)


def fit_linear(x, y, n=400000, seed=0):
    """Least-squares y ~ a x + b on a random subset of training pixels."""
    rng = np.random.default_rng(seed)
    x, y = x.reshape(-1), y.reshape(-1)
    k = rng.choice(len(x), min(n, len(x)), replace=False)
    a, b = np.polyfit(x[k], y[k], 1)
    return float(a), float(b)


def baseline_maps(data, split, train_store):
    """{name: (maps of the test layouts)} for the EDA estimates; calibration fitted on the training layouts."""
    y_tr = train_store.label_map
    out = {}
    egr_tr, egr_te = raw_maps(data, split['train'], 'egr'), raw_maps(data, split['test'], 'egr')
    rudy_tr, rudy_te = raw_maps(data, split['train'], 'rudy'), raw_maps(data, split['test'], 'rudy')
    out['eGR (EDA estimate)'] = egr_te
    a, b = fit_linear(egr_tr, y_tr)
    out['eGR (linear calibrated)'] = np.clip(a * egr_te + b, 0, 1)
    a, b = fit_linear(rudy_tr, y_tr)
    out['RUDY (linear calibrated)'] = np.clip(a * rudy_te + b, 0, 1)
    return out


# ---------------------------------------------------------------- one fold / all folds
def run_fold(data, test_design, ckpt_dir, out_dir, config=None, variants=VARIANTS, force=False, setting='aug'):
    from .experiments import make_stores
    cfg = {**CONFIG, **(config or {})}
    os.makedirs(out_dir, exist_ok=True)
    split, _, stores = make_stores(data, test_design, D.HOT_THRESHOLD, ckpt_dir)
    tr, va, te = stores['train'], stores['val'], stores['test']
    yv, yt = va.flat('y'), te.flat('y')
    t_egr = best_threshold(yv, va.flat('egr'))
    egr_lf1 = per_layout_f1(yt, te.flat('egr'), t_egr)

    patch_rows, pixel_rows, scores, vscores, maps = [], [], {}, {}, {}
    for name, use_params in variants.items():
        model, hist = train_map(name, use_params, tr, va, ckpt_dir, force=force, **cfg)
        vm, tm = predict_maps(model, va, amp=cfg['amp']), predict_maps(model, te, amp=cfg['amp'])
        maps[name] = tm
        vscores[name], scores[name] = patch_scores(vm), patch_scores(tm)
        thr = best_threshold(yv, vscores[name])
        lf1 = per_layout_f1(yt, scores[name], thr)
        patch_rows.append({'test_design': test_design, 'setting': setting, 'hotspot_rate': round(float(yt.mean()), 4),
                           'method': name, **{k: round(v, 4) for k, v in metrics(yt, scores[name], thr).items()},
                           'threshold (from val)': round(thr, 4), 'mean_layout_F1': round(float(lf1.mean()), 4),
                           'layouts_beating_eGR': int((lf1 > egr_lf1).sum()), 'layouts': len(lf1),
                           'train_minutes': hist['minutes'], 'best_epoch': hist['best_epoch']})
        pixel_rows.append({'test_design': test_design, 'method': name, **pixel_metrics(tm, te.label_map)})
    for name, m in baseline_maps(data, split, tr).items():
        pixel_rows.append({'test_design': test_design, 'method': name, **pixel_metrics(m, te.label_map)})
        maps[name] = m

    np.savez_compressed(os.path.join(out_dir, 'test_scores.npz'), y=yt, **scores)
    np.savez_compressed(os.path.join(out_dir, 'val_scores.npz'), y=yv, **vscores)
    pd.DataFrame(patch_rows).to_csv(os.path.join(out_dir, 'test_results.csv'), index=False)
    plot_maps(te, maps, os.path.join(out_dir, 'maps.png'))
    return patch_rows, pixel_rows


def cross_design(processed_dir, designs, ckpt_root='checkpoints_unet', out_root='outputs/unet', config=None, force=False):
    """Every design is the test design once. Writes all_folds.csv (patch level, same columns as
    outputs/lodo/all_folds.csv), pixel_metrics.csv and chosen.json (the variant used by the web app)."""
    data = D.load_designs(processed_dir, designs)
    patch_rows, pixel_rows = [], []
    for test in designs:
        print(f'\n=== test design: {test} ===')
        pr, px = run_fold(data, test, os.path.join(ckpt_root, test), os.path.join(out_root, test), config, force=force)
        patch_rows += pr; pixel_rows += px
    os.makedirs(out_root, exist_ok=True)
    patch, pixel = pd.DataFrame(patch_rows), pd.DataFrame(pixel_rows)
    patch.to_csv(os.path.join(out_root, 'all_folds.csv'), index=False)
    pixel.to_csv(os.path.join(out_root, 'pixel_metrics.csv'), index=False)
    mean_r = pixel[pixel.method.isin(VARIANTS)].groupby('method')['Pearson'].mean()
    chosen = {'variant': mean_r.idxmax(), 'mean_pixel_pearson': mean_r.round(4).to_dict(),
              'rule': 'higher mean pixel Pearson over the 3 test designs (fixed before training)'}
    with open(os.path.join(out_root, 'chosen.json'), 'w') as f:
        json.dump(chosen, f, indent=1)
    return patch, pixel, chosen


def summarize(patch, pixel):
    px = pixel.groupby('method', sort=False)[['Pearson', 'SSIM', 'NRMSE']].agg(['mean', 'std']).round(4)
    pa = patch.groupby('method', sort=False)[['ROC-AUC', 'PR-AUC', 'F1(hotspot)', 'Macro-F1', 'mean_layout_F1']].mean().round(4)
    pa['layouts_beating_eGR'] = patch.groupby('method', sort=False)['layouts_beating_eGR'].sum().astype(str) + ' / ' + \
        patch.groupby('method', sort=False)['layouts'].sum().astype(str)
    return px, pa


# ---------------------------------------------------------------- export to the web app
def export(test_design='Vortex-small', variant=None, ckpt_root='checkpoints_unet', out_root='outputs/unet',
           ckpt_dir='checkpoints', out_dir='outputs'):
    """Add the chosen variant of one fold to what the web app loads: checkpoints/UNet.pt + UNet.json,
    its validation threshold in thresholds.json, its scores in outputs/{test,val}_scores.npz and its row in
    outputs/test_results.csv. Requires the fold to be the one the other web models come from (same split)."""
    if variant is None:
        with open(os.path.join(out_root, 'chosen.json')) as f:
            variant = json.load(f)['variant']
    src_ck, src_out = os.path.join(ckpt_root, test_design), os.path.join(out_root, test_design)
    with open(os.path.join(src_ck, 'norm.json')) as f, open(os.path.join(ckpt_dir, 'norm.json')) as g:
        assert json.load(f) == json.load(g), 'normaliser differs: not the same split as the web models'
    stem = file_stem(variant)
    shutil.copy2(os.path.join(src_ck, f'{stem}.pt'), os.path.join(ckpt_dir, 'UNet.pt'))
    with open(os.path.join(src_ck, f'{stem}.json')) as f:
        meta = json.load(f)
    meta.update(name=variant, use_params=VARIANTS[variant])
    with open(os.path.join(ckpt_dir, 'UNet.json'), 'w') as f:
        json.dump(meta, f, indent=1)

    row = pd.read_csv(os.path.join(src_out, 'test_results.csv')).set_index('method').loc[variant]
    thr_path = os.path.join(ckpt_dir, 'thresholds.json')
    with open(thr_path) as f:
        thr = json.load(f)
    for v in VARIANTS:
        thr.pop(v, None)
    thr[variant] = float(row['threshold (from val)'])
    with open(thr_path, 'w') as f:
        json.dump(thr, f, indent=1)

    for kind in ('test', 'val'):
        web, new = dict(np.load(os.path.join(out_dir, f'{kind}_scores.npz'))), np.load(os.path.join(src_out, f'{kind}_scores.npz'))
        assert np.array_equal(web['y'], new['y']), f'{kind} labels differ: not the same layouts / order'
        for v in VARIANTS:
            web.pop(v, None)
        web[variant] = new[variant]
        np.savez_compressed(os.path.join(out_dir, f'{kind}_scores.npz'), **web)

    res = pd.read_csv(os.path.join(out_dir, 'test_results.csv'))
    res = res[~res['method'].isin(list(VARIANTS))]
    cols = [c for c in res.columns if c != 'method']
    res = pd.concat([res, pd.DataFrame([{'method': variant, **{c: row[c] for c in cols}}])], ignore_index=True)
    res.to_csv(os.path.join(out_dir, 'test_results.csv'), index=False)
    print(f'exported {variant} ({test_design}) -> {ckpt_dir}/UNet.pt, thresholds.json, {out_dir}/test_*')


# ---------------------------------------------------------------- figure
def plot_maps(store, maps, path, layouts=None):
    import matplotlib
    import matplotlib.pyplot as plt
    n = len(store.items)
    layouts = layouts or [0, n // 2, n - 1]
    cols = ['input RUDY', 'true utilisation', *maps]
    fig, ax = plt.subplots(len(layouts), len(cols), figsize=(2.4 * len(cols), 2.5 * len(layouts)), squeeze=False)
    for r, L in enumerate(layouts):
        ax[r, 0].imshow(store.X[L, 1], cmap='inferno')
        ax[r, 1].imshow(store.label_map[L], cmap='magma', vmin=0, vmax=1)
        for j, (k, m) in enumerate(maps.items()):
            ax[r, 2 + j].imshow(m[L], cmap='magma', vmin=0, vmax=1)
        for j, c in enumerate(cols):
            ax[r, j].set_title(c.replace(' (EDA estimate)', ''), fontsize=8)
            ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
    fig.tight_layout(); fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
