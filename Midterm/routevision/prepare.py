"""
Download CircuitNet-N14 routability features and convert them into compact
per-design .npz files (all maps resized to 256x256, float16).

The archive is read as a stream (tarfile 'r|gz'): only the needed feature maps
are decoded, so the ~10x larger extracted dataset never touches the disk.

Usage (local or Colab):
    python -m routevision.prepare --designs Vortex-small nvdla-small openc910-1
"""

import argparse
import io
import os
import re
import tarfile
import urllib.request

import cv2
import numpy as np

HF_URL = ('https://huggingface.co/datasets/CircuitNet/CircuitNet/resolve/main/'
          'CircuitNet-N14/routability_features/{design}.tar.gz')
SIZE = 256

# feature name -> directory inside the archive
FEATURES = {
    'cell_density': 'cell_density',
    'rudy': 'RUDY/RUDY',
    'rudy_pin': 'RUDY/RUDY_pin',
    'macro': 'macro_region',
    'gr_h': 'congestion/congestion_global_routing/utilization_based/congestion_GR_horizontal_util',
    'gr_v': 'congestion/congestion_global_routing/utilization_based/congestion_GR_vertical_util',
    'egr_h': 'congestion/congestion_early_global_routing/utilization_based/congestion_eGR_horizontal_util',
    'egr_v': 'congestion/congestion_early_global_routing/utilization_based/congestion_eGR_vertical_util',
}
INPUT_CHANNELS = ['cell_density', 'rudy', 'rudy_pin', 'macro']
PARAM_NAMES = ['freq', 'mp', 'fpu', 'fpa', 'p', 'fi']   # parsed from the sample file name


def parse_params(sample, design):
    """'Vortex-small_freq_200_mp_1_fpu_50_fpa_1.0_p_1_fi_ap' -> {'freq': 200.0, ...}."""
    rest = sample[len(design) + 1:]
    tokens = rest.split('_')
    out = {}
    for k, v in zip(tokens[0::2], tokens[1::2]):
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def params_vector(p):
    """Numeric design-parameter vector; the categorical 'fi' flag (ap / ar) becomes 0 / 1."""
    vec = []
    for k in PARAM_NAMES:
        v = p.get(k, 0.0)
        if k == 'fi':
            v = 1.0 if v == 'ar' else 0.0
        vec.append(float(v))
    return vec


def download(design, raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f'{design}.tar.gz')
    if not os.path.exists(path):
        url = HF_URL.format(design=design)
        print(f'[download] {url}')
        tmp = path + '.part'
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, path)
    return path


def _resize(a, nearest=False):
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    return cv2.resize(a.astype(np.float32), (SIZE, SIZE), interpolation=interp)


def convert(design, archive, out_dir):
    """Stream the archive once and write <out_dir>/<design>.npz."""
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f'{design}.npz')
    if os.path.exists(out):
        print(f'[skip] {out} exists')
        return out
    dir_to_feat = {f'{design}/{d}': k for k, d in FEATURES.items()}
    maps, shapes = {}, {}
    with tarfile.open(archive, 'r|gz') as tar:
        for m in tar:
            if not (m.isfile() and m.name.endswith('.npz')):
                continue
            name = m.name.lstrip('./')
            feat = dir_to_feat.get(os.path.dirname(name))
            if feat is None:
                continue
            sample = os.path.basename(name)[:-4]
            a = np.load(io.BytesIO(tar.extractfile(m).read()))['data']
            if feat == 'cell_density':
                shapes[sample] = a.shape
            maps.setdefault(sample, {})[feat] = _resize(a, nearest=(feat == 'macro')).astype(np.float16)
    samples = sorted(s for s, f in maps.items() if len(f) == len(FEATURES))
    dropped = len(maps) - len(samples)
    if dropped:
        print(f'[warn] {design}: {dropped} samples miss some feature maps and are dropped')
    X = np.stack([np.stack([maps[s][c] for c in INPUT_CHANNELS]) for s in samples])
    label = np.stack([np.maximum(maps[s]['gr_h'], maps[s]['gr_v']) for s in samples])
    egr = np.stack([np.maximum(maps[s]['egr_h'], maps[s]['egr_v']) for s in samples])
    params = np.array([params_vector(parse_params(s, design)) for s in samples], dtype=np.float32)
    np.savez_compressed(out, X=X, label=label, egr=egr, params=params, samples=np.array(samples),
                        orig_shape=np.array([shapes[s] for s in samples]), channels=np.array(INPUT_CHANNELS),
                        param_names=np.array(PARAM_NAMES))
    print(f'[done] {design}: {len(samples)} samples -> {out} ({os.path.getsize(out) / 1e6:.0f} MB)')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--designs', nargs='+', default=['Vortex-small', 'nvdla-small', 'openc910-1'])
    ap.add_argument('--raw_dir', default='data/raw')
    ap.add_argument('--out_dir', default='data/processed')
    args = ap.parse_args()
    for d in args.designs:
        convert(d, download(d, args.raw_dir), args.out_dir)


if __name__ == '__main__':
    main()
