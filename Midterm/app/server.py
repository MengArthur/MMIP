"""
RouteVision web app: Flask back end (REST API + model inference) and an
HTML/CSS/JS front end (templates/index.html, static/).

    python app/server.py            # http://127.0.0.1:5000  (waitress WSGI server)
    python app/server.py --dev      # Flask development server

On start-up the built-in examples are analysed in a background thread (all methods + Grad-CAM) and the
results are kept in demo/cache/, so during a live demo every built-in example opens instantly (ViT alone
needs ~20 s per layout on a laptop CPU). The cache key includes the checkpoint files, so retraining
invalidates it automatically.

API
    GET  /api/info      methods, demo layouts (thumbnails, warm-up status), test-set and
                        cross-design (leave-one-design-out) results
    GET  /api/status    which built-in examples are already pre-computed
    POST /api/predict   JSON {"demo": k}  or  multipart file=<layout .npz>
                        -> input images, hotspot scores of every method, ground truth, comparison,
                           ensemble disagreement (uncertainty) per patch
    POST /api/gradcam   JSON {"layout_id": ..., "method": "CNN" | "Multimodal"} -> Grad-CAM overlay
"""

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import threading
import time
from collections import OrderedDict

import numpy as np
from flask import Flask, jsonify, render_template, request
from matplotlib import colormaps
from PIL import Image
from werkzeug.exceptions import HTTPException

from inference import (CHANNELS, CKPT, METHODS, ROOT, BadLayout, Engine, describe_params,
                       load_demos, parse_npz)
from routevision import pixel as P
from routevision import uncertainty as U
from routevision.dataset import HOT_THRESHOLD, patch_means

MAX_UPLOAD_MB = 20
CACHE_SIZE = 16                                   # uploaded layouts kept in memory (demos are always kept)
DISK_CACHE = os.path.join(ROOT, 'demo', 'cache')
LODO_FOLDS = os.path.join(ROOT, 'outputs', 'lodo', 'all_folds.csv')
LODO_SETTING = os.path.join(ROOT, 'outputs', 'lodo', 'chosen_setting.json')
UNET_FOLDS = os.path.join(ROOT, 'outputs', 'unet', 'all_folds.csv')         # python -m routevision.pixel (Colab)
UNET_PIXEL = os.path.join(ROOT, 'outputs', 'unet', 'pixel_metrics.csv')
UNET_CHOSEN = os.path.join(ROOT, 'outputs', 'unet', 'chosen.json')
UNC_EVAL = os.path.join(ROOT, 'outputs', 'uncertainty', 'uncertainty.csv')   # python -m routevision.uncertainty
TEST_SCORES = os.path.join(ROOT, 'outputs', 'test_scores.npz')
UNC_TOP = 0.2               # "not sure" = among the 20 % most uncertain patches of the test design

# method -> (family shown in the UI, course week)
FAMILY = {
    'RUDY (EDA estimate)': ('EDA 粗估', '基準'), 'eGR (EDA estimate)': ('EDA 粗估', '基準'),
    'OpenCV + RandomForest': ('傳統 CV + ML', 'Week 1–2'), 'MLP': ('深度學習', 'Week 2'),
    'CNN': ('深度學習', 'Week 3'), 'Multimodal': ('深度學習', 'Week 2–3、5'), 'ViT': ('深度學習', 'Week 4'),
    'Ensemble': ('集成', '加分'), 'U-Net': ('深度學習', 'Week 3、加分'), 'U-Net (multimodal)': ('深度學習', 'Week 3、5、加分'),
}
DESIGN_TYPE = {'Vortex-small': 'GPU', 'nvdla-small': 'ML 加速器', 'openc910-1': 'RISC-V CPU'}


def _encode(pixels):
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def png(img, cmap='inferno', vmin=None, vmax=None):
    """2-D array -> colour-mapped PNG data URL."""
    a = np.asarray(img, np.float32)
    lo = float(a.min()) if vmin is None else vmin
    hi = float(a.max()) if vmax is None else vmax
    a = np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)
    return _encode((colormaps[cmap](a)[..., :3] * 255).astype(np.uint8))


def cam_overlay(base, cam, alpha=0.45):
    a = np.asarray(base, np.float32)
    g = np.clip(a / (a.max() + 1e-8), 0, 1)[..., None].repeat(3, -1)
    heat = colormaps['jet'](np.clip(cam, 0, 1))[..., :3]
    return _encode((((1 - alpha) * g + alpha * heat) * 255).astype(np.uint8))


def cam_heat(cam):
    """Grad-CAM as a transparent RGBA layer (opacity grows with attention), drawn over any base map."""
    c = np.clip(np.asarray(cam, np.float32), 0, 1)
    rgba = colormaps['jet'](c)
    rgba[..., 3] = np.clip(c ** 0.8 * 0.85, 0, 1)
    return _encode((rgba * 255).astype(np.uint8))


def model_fingerprint(ckpt=CKPT):
    """Names, sizes and modification times of the checkpoint files: changes whenever the models are retrained."""
    if not os.path.isdir(ckpt):
        return 'none'
    return ';'.join(f'{f}:{os.path.getsize(os.path.join(ckpt, f))}:{int(os.path.getmtime(os.path.join(ckpt, f)))}'
                    for f in sorted(os.listdir(ckpt)))


def read_lodo(path=LODO_FOLDS, setting_path=LODO_SETTING, extra=(UNET_FOLDS,)):
    """Leave-one-design-out results (outputs/lodo/, outputs/unet/) of the setting the web app's models come from."""
    if not os.path.exists(path):
        return None
    setting = 'aug'
    if os.path.exists(setting_path):
        with open(setting_path) as f:
            setting = json.load(f).get('setting', setting)
    records, rows, designs = [], [], {}
    for p in (path, *[e for e in extra if os.path.exists(e)]):                # + the U-Net folds, same columns
        with open(p, encoding='utf-8') as f:
            records += list(csv.DictReader(f))
    for r in records:
        if r['setting'] != setting:
            continue
        designs[r['test_design']] = float(r['hotspot_rate'])
        rows.append({'design': r['test_design'], 'method': r['method'],
                     **{k: float(r[k]) for k in ('ROC-AUC', 'PR-AUC', 'Macro-F1', 'F1(hotspot)')},
                     'wins': None if r['method'].startswith('eGR') else int(r['layouts_beating_eGR']),
                     'layouts': int(r['layouts'])})
    return {'setting': setting, 'rows': rows,
            'designs': [{'name': d, 'type': DESIGN_TYPE.get(d, ''), 'hotspot_rate': h} for d, h in designs.items()]}


def read_pixel(path=UNET_PIXEL, chosen=UNET_CHOSEN):
    """Pixel-level results of the U-Net folds (routevision/pixel.py), None if they have not been run."""
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        rows = [{k: (v if k in ('test_design', 'method') else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]
    ch = None
    if os.path.exists(chosen):
        with open(chosen) as f:
            ch = json.load(f)
    return {'rows': rows, 'chosen': ch}


def util_maps(entry):
    """Full-resolution utilisation maps (same colour scale 0-1) and this layout's pixel agreement with the truth."""
    store, maps = entry['store'], entry.get('maps') or {}
    if not maps:
        return {'maps': None, 'map_metrics': None}
    cm = lambda a: png(a, cmap='magma', vmin=0, vmax=1)
    raw_rudy = np.asarray(store.X[0, 1], np.float32)
    out = {name: cm(m) for name, m in maps.items()}
    if entry['has_egr']:
        out['eGR (EDA estimate)'] = cm(entry['egr_map'])
    out['RUDY (EDA estimate)'] = png(raw_rudy, cmap='magma', vmin=0, vmax=float(np.percentile(raw_rudy, 99.5)) + 1e-6)
    metrics = None
    if entry['has_label']:
        t = store.label_map[:1]
        cands = {**maps, **({'eGR (EDA estimate)': entry['egr_map']} if entry['has_egr'] else {}),
                 'RUDY (EDA estimate)': raw_rudy}
        metrics = {k: {'Pearson': round(float(P.pearson(np.asarray(v)[None], t)[0]), 3),
                       'SSIM': None if k.startswith('RUDY') else round(float(P.ssim(np.asarray(v)[None], t)[0]), 3)}
                   for k, v in cands.items()}
        out['truth'] = cm(store.label_map[0])
    return {'maps': out, 'map_metrics': metrics}


def read_uncertainty_eval(path=UNC_EVAL):
    """Cross-design evaluation of the uncertainty (routevision/uncertainty.py), None if it has not been run."""
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as f:
        return [{c: (v if c in ('test_design', 'measure') else (float(v) if v != '' else None)) for c, v in r.items()}
                for r in csv.DictReader(f)]


def uncertainty_cutoff(path=TEST_SCORES, top=UNC_TOP):
    """Disagreement above which a patch is among the `top` most uncertain patches of the test design
    (from the unlabelled test scores only)."""
    if not os.path.exists(path):
        return None
    z = np.load(path)
    if not all(m in z.files for m in U.MEMBERS):
        return None
    return float(np.quantile(U.disagreement(z), 1 - top))


def create_app(engine=None, warm=False, disk_cache=True):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = MAX_UPLOAD_MB * 1024 * 1024
    app.json.ensure_ascii = False
    app.json.sort_keys = False                  # keep design-parameter order (freq, mp, fpu, ...)
    engine = engine or Engine()
    demos = load_demos()
    fingerprint = model_fingerprint()
    lodo = read_lodo()
    unc_eval = read_uncertainty_eval()
    pixel = read_pixel()
    unc_cut = uncertainty_cutoff()
    pinned = {}                                 # demo-k -> analysed layout (never evicted)
    cache = OrderedDict()                       # up-<sha1> -> analysed layout (LRU)
    lock = threading.RLock()                    # one analysis at a time; a request waits for the warm-up item

    demo_meta = []
    for name, layout, group in demos:
        design, sample = name.split(' | ')
        n_hot = int((patch_means(np.asarray(layout['label'], np.float32)[None])[0] >= HOT_THRESHOLD).sum())
        demo_meta.append({'name': name, 'group': group, 'design': design, 'sample': sample.replace(design + '_', ''),
                          'params': describe_params(layout['params']), 'n_hot': n_hot,
                          'thumb': png(np.asarray(layout['X'][1], np.float32)[::4, ::4])})    # RUDY, 64x64

    # ------------------------------------------------------------ disk cache (demo/cache/<key>.npz)
    def disk_key(layout, has_egr):
        h = hashlib.sha1(fingerprint.encode())
        for k in ('X', 'params', 'label', 'egr'):
            h.update(np.ascontiguousarray(layout[k]).tobytes())
        h.update(b'egr' if has_egr else b'-')
        return h.hexdigest()[:24]

    def disk_load(key):
        p = os.path.join(DISK_CACHE, key + '.npz')
        if not disk_cache or not os.path.exists(p):
            return None, {}, {}
        try:
            z = np.load(p, allow_pickle=False)
            return ({k[3:]: z[k].astype(np.float32) for k in z.files if k.startswith('p__')},
                    {k[3:]: z[k].astype(np.float32) for k in z.files if k.startswith('c__')},
                    {k[3:]: z[k].astype(np.float32) for k in z.files if k.startswith('m__')})
        except Exception:                                        # unreadable file: recompute
            return None, {}, {}

    def disk_save(key, probs, cams, maps):
        if not disk_cache:
            return
        os.makedirs(DISK_CACHE, exist_ok=True)
        tmp = os.path.join(DISK_CACHE, key + '.tmp.npz')
        np.savez_compressed(tmp, **{'p__' + m: v for m, v in probs.items()},
                            **{'c__' + m: v.astype(np.float16) for m, v in cams.items()},
                            **{'m__' + m: v.astype(np.float16) for m, v in maps.items()})
        os.replace(tmp, os.path.join(DISK_CACHE, key + '.npz'))

    # ------------------------------------------------------------ analysis
    def analyse(layout_id, name, source, layout, has_label, has_egr):
        mem = pinned if source == 'demo' else cache
        if layout_id in pinned:                                  # ready demo: don't wait for the warm-up thread
            return pinned[layout_id], 0.0
        with lock:
            if layout_id in mem:
                if mem is cache:
                    cache.move_to_end(layout_id)
                return mem[layout_id], 0.0
            t0 = time.perf_counter()
            store = engine.store(layout, has_egr)
            key = disk_key(layout, has_egr)
            probs, cams, maps = disk_load(key)
            expected = {m for m in engine.available() if has_egr or m != 'eGR (EDA estimate)'}
            if probs is None or set(probs) != expected or (engine.unet_name and engine.unet_name not in maps):
                maps = {}
                probs, cams = engine.predict_all(store, maps), {}
                disk_save(key, probs, cams, maps)
            entry = {'name': name, 'source': source, 'store': store, 'has_label': has_label, 'has_egr': has_egr,
                     'params': layout['params'], 'probs': probs, 'cams': cams, 'maps': maps, 'key': key,
                     'egr_map': np.asarray(layout['egr'], np.float32)}
            mem[layout_id] = entry
            while len(cache) > CACHE_SIZE:
                cache.popitem(last=False)
            return entry, time.perf_counter() - t0

    def get_cam(entry, method):
        if method in entry['cams']:
            return entry['cams'][method]
        with lock:
            if method not in entry['cams']:
                entry['cams'][method] = engine.gradcam(method, entry['store'])
                disk_save(entry['key'], entry['probs'], entry['cams'], entry['maps'])
            return entry['cams'][method]

    def ready_demos():
        cams = engine.cam_methods()
        return [k for k in range(len(demos))
                if f'demo-{k}' in pinned and all(m in pinned[f'demo-{k}']['cams'] for m in cams)]

    def warm_up():
        t0 = time.perf_counter()
        for k, (name, layout, _) in enumerate(demos):
            try:
                entry, _ = analyse(f'demo-{k}', name, 'demo', layout, True, True)
                for m in engine.cam_methods():
                    get_cam(entry, m)
            except Exception as e:                               # never take the server down
                print(f'warm-up of demo {k} failed: {e}', flush=True)
        print(f'warm-up done in {time.perf_counter() - t0:.0f} s: '
              f'{len(ready_demos())}/{len(demos)} built-in examples ready', flush=True)

    if warm:
        threading.Thread(target=warm_up, daemon=True).start()

    # ------------------------------------------------------------ routes
    @app.errorhandler(HTTPException)
    def http_error(e):
        if request.path.startswith('/api/'):
            msg = f'檔案超過 {MAX_UPLOAD_MB} MB' if e.code == 413 else e.description
            return jsonify(error=msg), e.code
        return e

    @app.get('/')
    def index():
        return render_template('index.html')

    @app.get('/api/info')
    def info():
        avail = engine.available()
        methods = [{'name': m, 'subtitle': sub, 'description': desc, 'available': m in avail,
                    'family': FAMILY.get(m, ('', ''))[0], 'course': FAMILY.get(m, ('', ''))[1],
                    'gradcam': m in engine.cam_methods(),
                    'threshold': engine.thr.get(m), 'test': engine.test_results.get(m)}
                   for m, (sub, desc) in METHODS.items() if m in avail]
        ready = set(ready_demos())
        return jsonify(methods=methods, demos=[{**d, 'ready': k in ready} for k, d in enumerate(demo_meta)],
                       overall=engine.overall, lodo=lodo, pixel=pixel,
                       uncertainty={'members': U.MEMBERS, 'cutoff': unc_cut, 'top': UNC_TOP, 'eval': unc_eval}
                       if unc_cut else None,
                       channels=CHANNELS,
                       grid=8, patch=32, size=256, hot_threshold=HOT_THRESHOLD)

    @app.get('/api/status')
    def status():
        return jsonify(ready=ready_demos(), total=len(demos))

    @app.post('/api/predict')
    def predict():
        if 'file' in request.files:
            f = request.files['file']
            raw = f.read()
            if not raw:
                return jsonify(error='上傳的檔案是空的'), 400
            try:
                layout, has_label, has_egr = parse_npz(raw)
            except BadLayout as e:
                return jsonify(error=str(e)), 400
            layout_id = 'up-' + hashlib.sha1(raw).hexdigest()[:16]
            entry, dt = analyse(layout_id, f.filename or 'upload.npz', 'upload', layout, has_label, has_egr)
            group = '上傳的版圖'
        else:
            body = request.get_json(silent=True) or {}
            k = body.get('demo')
            if type(k) is not int or not 0 <= k < len(demos):
                return jsonify(error=f'demo 必須是 0–{len(demos) - 1} 的整數'), 400
            name, layout, group = demos[k]
            layout_id = f'demo-{k}'
            entry, dt = analyse(layout_id, name, 'demo', layout, True, True)

        store, probs = entry['store'], entry['probs']
        X = store.X[0]
        out = {
            'layout_id': layout_id,
            'demo_index': int(layout_id.split('-')[1]) if entry['source'] == 'demo' else None,
            'name': entry['name'], 'source': entry['source'], 'group': group,
            'has_label': entry['has_label'], 'has_egr': entry['has_egr'],
            'params': describe_params(entry['params']),
            'channels': [{'title': t + ('（全為 0：此版圖沒有 macro）' if c == 3 and not X[c].any() else ''),
                          'src': png(X[c]), 'gray': png(X[c], cmap='gray'), 'empty': not X[c].any()}
                         for c, t in enumerate(CHANNELS)],
            'base': png(X[1], cmap='gray'),
            'scores': {m: v.round(4).tolist() for m, v in probs.items()},
            'thresholds': {m: engine.thr.get(m, 0.5) for m in probs},
            'truth': store.y[0].astype(int).tolist() if entry['has_label'] else None,
            'util': store.util[0].round(3).tolist() if entry['has_label'] else None,
            'truth_map': png(store.label_map[0], cmap='Reds', vmin=0, vmax=1) if entry['has_label'] else None,
            **util_maps(entry),
            'comparison': engine.compare(probs, store.y[0]) if entry['has_label'] else None,
            'uncertainty': (U.disagreement(probs).round(4).tolist()
                            if unc_cut is not None and all(m in probs for m in U.MEMBERS) else None),
            'elapsed_ms': round(dt * 1000),
        }
        return jsonify(out)

    @app.post('/api/gradcam')
    def gradcam():
        body = request.get_json(silent=True) or {}
        lid = body.get('layout_id')
        entry = pinned.get(lid) or cache.get(lid)
        method = body.get('method')
        if entry is None:
            return jsonify(error='找不到這個版圖，請重新分析一次'), 404
        if method not in engine.cam_methods():
            return jsonify(error=f'Grad-CAM 只支援 {", ".join(engine.cam_methods())}'), 400
        t0 = time.perf_counter()
        cam = get_cam(entry, method)
        return jsonify(method=method, overlay=cam_overlay(entry['store'].X[0, 0], cam), heat=cam_heat(cam),
                       elapsed_ms=round((time.perf_counter() - t0) * 1000))

    return app


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='RouteVision Flask web app')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=int(os.environ.get('PORT', 5000)))
    ap.add_argument('--dev', action='store_true', help="use Flask's development server instead of waitress")
    ap.add_argument('--no-warm', action='store_true', help='do not pre-compute the built-in examples at start-up')
    args = ap.parse_args()
    app = create_app(warm=not args.no_warm)
    if args.dev:
        app.run(host=args.host, port=args.port, debug=False, threaded=True)
    else:
        # waitress: production WSGI server that runs on Windows; Flask's development server
        # intermittently reset connections and stalled for ~20 s on this machine
        from waitress import serve
        print(f'RouteVision running on http://{args.host}:{args.port}  (Ctrl+C to stop)', flush=True)
        serve(app, host=args.host, port=args.port, threads=4)
