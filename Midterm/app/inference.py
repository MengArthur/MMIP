"""
Inference engine shared by the Flask web app (app/server.py) and the Streamlit
fallback (app/streamlit_app.py): loads every trained method once, scores all
64 patches of a layout with each method, and computes Grad-CAM.
"""

import csv
import io
import json
import os
import sys
import threading

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from routevision import dataset as D
from routevision.features import store_features
from routevision.gradcam import layout_cam, unet_cam
from routevision.models import build, UNet
from routevision.train import per_layout_f1

CKPT = os.path.join(ROOT, 'checkpoints')
DEMO = os.path.join(ROOT, 'demo', 'demo_layouts.npz')
RESULTS = os.path.join(ROOT, 'outputs', 'test_results.csv')
SCORES = os.path.join(ROOT, 'outputs', 'test_scores.npz')
DEEP = ['MLP', 'CNN', 'Multimodal', 'ViT']
CAM_MODELS = ['CNN', 'Multimodal']
CHANNELS = ['cell 密度', '繞線需求 RUDY', '腳位繞線需求', 'macro 區域']
PARAM_NAMES = ['freq', 'mp', 'fpu', 'fpa', 'p', 'fi']

# name -> (sidebar subtitle, description)
METHODS = {
    'RUDY (EDA estimate)': ('EDA 粗估', 'EDA 工具的快速估計：把每條 net 的線長平均攤在它的範圍內。不需學習，作為基準。'),
    'eGR (EDA estimate)': ('EDA 粗估', 'EDA 工具的 early global routing：快速粗略地繞一次線。比 RUDY 準，但仍和最終結果有落差。'),
    'OpenCV + RandomForest': ('傳統 CV + ML', '傳統方法：OpenCV 擷取密度、梯度等手工特徵，再用 Random Forest 分類。'),
    'MLP': ('多層感知器', '把影像縮小後攤平，用多層感知器分類。'),
    'CNN': ('卷積神經網路', '卷積神經網路，直接從版圖影像學習壅塞的空間樣式。'),
    'Multimodal': ('影像 + 設計參數', '多模態：影像 CNN ＋ 設計參數（時脈、floorplan 使用率…）與區塊位置的 MLP，兩者融合後分類。'),
    'ViT': ('Vision Transformer', 'ImageNet 預訓練的 Vision Transformer，改成 4 通道輸入後微調。'),
    'Ensemble': ('RF + Multimodal + ViT', '集成：Random Forest、Multimodal、ViT 三者的熱點機率取平均，不需額外訓練；閾值同樣在驗證集上決定。'),
    'U-Net': ('逐像素預測使用率', 'U-Net（ImageNet ResNet-18 編碼器）直接預測整張 256×256 繞線使用率圖，再取各區塊平均判斷熱點。'),
    'U-Net (multimodal)': ('逐像素＋設計參數', 'U-Net（ImageNet ResNet-18 編碼器）預測整張繞線使用率圖；設計參數以 FiLM 調整瓶頸層特徵。再取各區塊平均判斷熱點。'),
}


class BadLayout(ValueError):
    """Uploaded file is not a valid single-layout .npz."""


def _squeeze(a, ndim, name):
    a = np.asarray(a)
    if a.ndim == ndim + 1 and a.shape[0] == 1:
        a = a[0]
    if a.ndim != ndim:
        raise BadLayout(f'{name} 的維度應為 {ndim}，實際為 {a.shape}')
    return a


def parse_npz(raw):
    """Bytes of an .npz (X: 4x256x256, params: 6; label, egr optional) -> (layout, has_label, has_egr)."""
    try:
        z = np.load(io.BytesIO(raw), allow_pickle=False)
        files = set(z.files)
    except Exception as e:                                # not a zip / corrupted
        raise BadLayout(f'無法讀取 .npz：{e}') from e
    for k in ('X', 'params'):
        if k not in files:
            raise BadLayout(f'缺少欄位 {k}')
    X = _squeeze(z['X'], 3, 'X')
    params = _squeeze(z['params'], 1, 'params')
    if X.shape != (4, D.SIZE, D.SIZE):
        raise BadLayout(f'X 應為 4×{D.SIZE}×{D.SIZE}，實際為 {X.shape}')
    if params.shape != (6,):
        raise BadLayout(f'params 應有 6 個值，實際為 {params.shape}')
    lay = {'X': X.astype(np.float16), 'params': params.astype(np.float32)}
    has = {}
    for k in ('label', 'egr'):
        has[k] = k in files
        m = _squeeze(z[k], 2, k) if has[k] else np.zeros((D.SIZE, D.SIZE), np.float16)
        if m.shape != (D.SIZE, D.SIZE):
            raise BadLayout(f'{k} 應為 {D.SIZE}×{D.SIZE}，實際為 {m.shape}')
        lay[k] = m
    if not np.isfinite(lay['X']).all():
        raise BadLayout('X 含有 NaN 或無限大')
    return lay, has['label'], has['egr']


def read_test_results(path=RESULTS):
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return {r['method']: {k: float(v) for k, v in r.items() if k != 'method'} for r in csv.DictReader(f)}


class Engine:
    def __init__(self, ckpt=CKPT):
        if not os.path.exists(os.path.join(ckpt, 'norm.json')):
            raise FileNotFoundError('找不到 checkpoints/norm.json：請先在 Colab 執行訓練 notebook，'
                                    '並把 RouteVision_results.zip 解壓到專案根目錄。')
        self.norm = D.Normalizer.from_json(os.path.join(ckpt, 'norm.json'))
        with open(os.path.join(ckpt, 'thresholds.json')) as f:
            self.thr = json.load(f)
        self.models = {}
        for m in DEEP:
            p = os.path.join(ckpt, f'{m}.pt')
            if os.path.exists(p):
                net = build(m, pretrained=False) if m == 'ViT' else build(m)
                net.load_state_dict(torch.load(p, map_location='cpu'))
                self.models[m] = net.eval()
        self.unet_name = None                              # pixel-level model, routevision/pixel.py
        if os.path.exists(os.path.join(ckpt, 'UNet.pt')) and os.path.exists(os.path.join(ckpt, 'UNet.json')):
            with open(os.path.join(ckpt, 'UNet.json')) as f:
                meta = json.load(f)
            u = UNet(use_params=meta['use_params'], pretrained=False)
            u.load_state_dict(torch.load(os.path.join(ckpt, 'UNet.pt'), map_location='cpu'))
            self.models['UNet'], self.unet_name = u.eval(), meta['name']
        self.rf = None
        if os.path.exists(os.path.join(ckpt, 'RandomForest.joblib')):
            import joblib
            self.rf = joblib.load(os.path.join(ckpt, 'RandomForest.joblib'))
        self.ensemble = []                                 # members, written by experiments.run (ensemble.json)
        if os.path.exists(os.path.join(ckpt, 'ensemble.json')) and 'Ensemble' in self.thr:
            with open(os.path.join(ckpt, 'ensemble.json')) as f:
                self.ensemble = json.load(f)['members']
        self.test_results = read_test_results()
        self.overall = overall_stats(self.thr, self.test_results)
        self.lock = threading.Lock()                       # one inference at a time (CPU-bound anyway)

    def available(self):
        """Methods that can run with the loaded checkpoints (eGR also needs the layout's egr map)."""
        out = ['RUDY (EDA estimate)', 'eGR (EDA estimate)']
        if self.rf is not None:
            out.append('OpenCV + RandomForest')
        out += [m for m in DEEP if m in self.models]
        if self.ensemble and all(m in out for m in self.ensemble):
            out.append('Ensemble')
        if self.unet_name:
            out.append(self.unet_name)
        return out

    def store(self, layout, has_egr=True):
        data = {'one': {k: v[None] for k, v in layout.items()}}
        s = D.LayoutStore(data, [('one', 0)], self.norm, threshold=D.HOT_THRESHOLD)
        s.has_egr = has_egr
        return s

    @torch.no_grad()
    def _deep(self, model, store):
        xs = torch.from_numpy(np.stack([store.window(0, r, c) for r in range(D.GRID) for c in range(D.GRID)]))
        side = torch.from_numpy(np.stack([store.side(0, r, c) for r in range(D.GRID) for c in range(D.GRID)]))
        return torch.sigmoid(model(xs, side)).numpy().reshape(D.GRID, D.GRID)

    @torch.no_grad()
    def unet_map(self, store):
        """(256, 256) predicted final routing utilisation."""
        x = torch.from_numpy(np.ascontiguousarray(store.X[:1]))
        return self.models['UNet'](x, torch.from_numpy(store.params[:1])).numpy()[0]

    def predict_all(self, store, maps=None):
        """{method: (8, 8) hotspot score}; thresholds for each method are in self.thr.
        If a dict is passed as `maps`, the U-Net's full utilisation map is put in it."""
        with self.lock:
            out = {'RUDY (EDA estimate)': store.rudy[0]}
            if store.has_egr:
                out['eGR (EDA estimate)'] = store.egr[0]
            if self.rf is not None:
                out['OpenCV + RandomForest'] = self.rf.predict_proba(store_features(store))[:, 1].reshape(D.GRID, D.GRID)
            for m in DEEP:
                if m in self.models:
                    out[m] = self._deep(self.models[m], store)
            if self.unet_name:
                pm = self.unet_map(store)
                out[self.unet_name] = D.patch_means(pm[None])[0]
                if maps is not None:
                    maps[self.unet_name] = pm.astype(np.float32)
            if self.ensemble and all(m in out for m in self.ensemble):
                out['Ensemble'] = np.mean([out[m] for m in self.ensemble], 0)
        return {k: np.asarray(v, np.float32) for k, v in out.items()}

    def cam_methods(self):
        """Methods with a Grad-CAM: the patch CNNs (Grad-CAM) and the U-Net (Seg-Grad-CAM)."""
        return [m for m in CAM_MODELS if m in self.models] + ([self.unet_name] if self.unet_name else [])

    def gradcam(self, method, store):
        if method not in self.cam_methods():
            raise KeyError(method)
        with self.lock:
            if method == self.unet_name:
                return unet_cam(self.models['UNet'], store.X[0], store.params[0], self.thr.get(method, 0.5))
            cam, _ = layout_cam(self.models[method], store, 0)
        return cam

    def compare(self, probs, truth):
        """Per-layout hits / false alarms / F1 of every method against the ground-truth patches."""
        y = truth.reshape(-1).astype(bool)
        rows = []
        for m, s in probs.items():
            pr = s.reshape(-1) >= self.thr.get(m, 0.5)
            tp, fp, fn = int((pr & y).sum()), int((pr & ~y).sum()), int((~pr & y).sum())
            rows.append({'method': m, 'tp': tp, 'fp': fp, 'fn': fn, 'n_true': int(y.sum()),
                         'n_pred': int(pr.sum()), 'f1': round(2 * tp / max(2 * tp + fp + fn, 1), 3)})
        return rows


def load_demos(path=DEMO):
    """[(name, layout dict, group)] from demo/demo_layouts.npz (built by app/make_demo_set.py)."""
    if not os.path.exists(path):
        return []
    z = np.load(path, allow_pickle=False)
    groups = z['groups'] if 'groups' in z.files else ['測試設計（沒參與訓練）'] * len(z['samples'])
    return [(f'{d} | {s}', {k: z[k][i] for k in ('X', 'label', 'egr', 'params')}, str(g))
            for i, (d, s, g) in enumerate(zip(z['designs'], z['samples'], groups))]


def overall_stats(thr, results, scores_path=SCORES):
    """Whole-test-set comparison: metrics from outputs/test_results.csv plus, per method, the mean
    per-layout F1 and in how many test layouts it beats eGR (from outputs/test_scores.npz)."""
    if not os.path.exists(scores_path):
        return None
    z = np.load(scores_path, allow_pickle=False)
    n = len(z['y']) // (D.GRID * D.GRID)
    f1 = lambda m: per_layout_f1(z['y'], z[m], thr.get(m, 0.5))
    ref = f1('eGR (EDA estimate)') if 'eGR (EDA estimate)' in z.files else None
    rows = []
    for m in z.files:
        if m == 'y':
            continue
        f = f1(m)
        r = {'method': m, 'mean_layout_f1': round(float(f.mean()), 4), **(results.get(m) or {})}
        if ref is not None and m != 'eGR (EDA estimate)':
            r.update(wins_vs_egr=int((f > ref).sum()), ties_vs_egr=int((f == ref).sum()), losses_vs_egr=int((f < ref).sum()))
        rows.append(r)
    return {'n_layouts': n, 'n_patches': int(len(z['y'])), 'rows': rows}


def describe_params(params):
    out = {}
    for k, v in zip(PARAM_NAMES, np.asarray(params, float)):
        out[k] = ('ar' if v >= 0.5 else 'ap') if k == 'fi' else (int(v) if float(v).is_integer() else round(float(v), 3))
    return out
