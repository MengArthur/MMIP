"""
Build demo/demo_layouts.npz: the layouts offered as built-in examples in the web app.

The examples are grouped and labelled so the demo stays honest, and chosen from the saved test scores
(outputs/test_scores.npz, checkpoints/thresholds.json), so the labels stay correct after retraining:
  1. test design, the learned models clearly beat the EDA estimates
     (largest margin of min(RF, Multimodal, ViT) per-layout F1 over max(RUDY, eGR))
  2. test design, eGR beats the multimodal model (largest margin)
  3. validation layouts of the training designs (same design, layout itself not trained on; these designs
     have macros, Vortex-small has none). Picked by hotspot count only, not by model performance.
Layouts with fewer than MIN_HOT hotspot patches are skipped (their F1 is too noisy to be an example), and
the fi=ap / fi=ar twins of an already chosen layout are skipped. The page also shows the whole-test-set
metrics, so a single example never stands for the result.

    python app/make_demo_set.py
"""

import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from routevision import dataset as D
from routevision.train import per_layout_f1

TEST = 'Vortex-small'
DESIGNS = [TEST, 'nvdla-small', 'openc910-1']
LEARNED = ['OpenCV + RandomForest', 'Multimodal', 'ViT']
EDA = ['RUDY (EDA estimate)', 'eGR (EDA estimate)']
MIN_HOT = 8
GOOD, BAD = '測試設計 Vortex-small（沒參與訓練）｜模型表現好的例子', '測試設計 Vortex-small（沒參與訓練）｜eGR 比較準的例子'
VAL = '訓練設計的驗證版圖（同設計但此版圖沒參與訓練，有 macro）'
VAL_PICKS = [('nvdla-small', 'freq_200_mp_1_fpu_65_fpa_1.0_p_1_fi_ap'),     # 12 hotspot patches
             ('openc910-1', 'freq_500_mp_1_fpu_50_fpa_1.0_p_4_fi_ar')]      # 34, the fewest in its validation set


def pick(order, names, n, taken):
    out = []
    for k in order:
        twin = re.sub(r'_fi_a[pr]$', '', names[k])
        if twin not in taken:
            out.append(k); taken.add(twin)
        if len(out) == n:
            break
    return out


def main(processed=os.path.join(ROOT, 'data', 'processed'), scores=os.path.join(ROOT, 'outputs', 'test_scores.npz'),
         thresholds=os.path.join(ROOT, 'checkpoints', 'thresholds.json'), out=os.path.join(ROOT, 'demo', 'demo_layouts.npz')):
    data = D.load_designs(processed, DESIGNS)
    split = D.make_split(data, TEST)
    test = split['test']
    names = [str(data[d]['samples'][i]) for d, i in test]
    z, thr = np.load(scores), json.load(open(thresholds))
    assert len(z['y']) == len(test) * D.GRID * D.GRID, 'test_scores.npz is not from the Vortex-small fold'
    F = {m: per_layout_f1(z['y'], z[m], thr[m]) for m in LEARNED + EDA if m in z.files}
    hot = z['y'].reshape(len(test), -1).sum(1)
    ok = hot >= MIN_HOT
    learned = [m for m in LEARNED if m in F]
    margin = np.min([F[m] for m in learned], 0) - np.max([F[m] for m in EDA], 0)
    gap = F['eGR (EDA estimate)'] - F['Multimodal']
    taken = set()
    good = pick([k for k in np.argsort(-margin) if ok[k] and margin[k] > 0], names, 3, taken)
    bad = pick([k for k in np.argsort(-gap) if ok[k] and gap[k] > 0], names, 2, taken)

    picks, groups = [], []
    for g, ks in ((GOOD, good), (BAD, bad)):
        for k in ks:
            picks.append(test[k]); groups.append(g)
            print(f'{g[-10:]} {names[k]}  hot {hot[k]}  ' + '  '.join(f'{m.split()[0]} {F[m][k]:.3f}' for m in EDA + learned))
    val = {(d, str(data[d]['samples'][i])): (d, i) for d, i in split['val']}
    for d, s in VAL_PICKS:
        assert (d, f'{d}_{s}') in val, f'{d}_{s} is not a validation layout'
        picks.append(val[(d, f'{d}_{s}')]); groups.append(VAL)
    np.savez_compressed(out,
                        X=np.stack([data[d]['X'][i] for d, i in picks]),
                        label=np.stack([data[d]['label'][i] for d, i in picks]),
                        egr=np.stack([data[d]['egr'][i] for d, i in picks]),
                        params=np.stack([data[d]['params'][i] for d, i in picks]),
                        samples=np.array([str(data[d]['samples'][i]) for d, i in picks]),
                        designs=np.array([d for d, _ in picks]), groups=np.array(groups))
    print(f'{len(picks)} layouts -> {out}')


if __name__ == '__main__':
    main()
