"""
MMIP HW05: Multimodal Deep Learning -- one-command pipeline (Quiz 1-3).

Trained models are cached in Checkpoints/ and extracted features in Data/, so a second
run only re-evaluates. The first run needs a GPU (about 1 hour on a Colab T4).
"""

import os
import sys
import time

import torch

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from Code.common import DEVICE
from Code.data import load_cub, load_flickr8k
from Code.quiz1_retrieval import run_quiz1
from Code.quiz2_multimodal import run_quiz2
from Code.quiz3_clip_caption import run_quiz3


def main():
    t0 = time.perf_counter()
    data_dir, ckpt_dir, out_dir = (os.path.join(BASE, d) for d in ('Data', 'Checkpoints', 'Output'))
    for d in (data_dir, ckpt_dir, out_dir):
        os.makedirs(d, exist_ok=True)
    gpu = f' ({torch.cuda.get_device_name(0)})' if DEVICE.type == 'cuda' else ''
    print(f'PyTorch {torch.__version__} | device: {DEVICE}{gpu}')
    q1 = run_quiz1(load_cub(data_dir), ckpt_dir, out_dir)
    flickr = load_flickr8k(data_dir)
    q2 = run_quiz2(flickr, data_dir, ckpt_dir, out_dir)
    q3 = run_quiz3(flickr, data_dir, ckpt_dir, out_dir)
    print('\n' + '=' * 70 + '\nSUMMARY\n' + '=' * 70)
    for t in (q1['table'], q2['table'], q3['delta']):
        print(t.to_string(index=False), end='\n\n')
    print(f'Total time: {(time.perf_counter() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
