"""
MMIP HW04: Time-Series Models & Transformers -- one-command pipeline.

Runs Quiz 1-4 end to end. Trained models are cached in Checkpoints/ and Gemini
responses in Output/quiz4_gemini_cache.json, so a second run only regenerates
the evaluation and figures. The first run needs a GPU (about 1 hour on a Colab T4).
Quiz 4 needs a Gemini API key in the GEMINI_API_KEY environment variable.
"""

import os
import sys
import time

import torch

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from Code.common import DEVICE
from Code.data import load_imdb, load_pets, load_flickr8k
from Code.quiz1_datasets import run_quiz1
from Code.quiz2_sentiment import run_quiz2, MAX_LEN
from Code.quiz3_vit import run_quiz3
from Code.quiz4_captioning import run_quiz4


def main():
    t0 = time.perf_counter()
    data_dir, ckpt_dir, out_dir = (os.path.join(BASE_DIR, d) for d in ('Data', 'Checkpoints', 'Output'))
    for d in (data_dir, ckpt_dir, out_dir):
        os.makedirs(d, exist_ok=True)
    gpu = f' ({torch.cuda.get_device_name(0)})' if DEVICE.type == 'cuda' else ''
    print(f'PyTorch {torch.__version__} | device: {DEVICE}{gpu}')

    imdb, pets, flickr = load_imdb(data_dir), load_pets(data_dir), load_flickr8k(data_dir)
    run_quiz1(imdb, pets, flickr, out_dir, MAX_LEN)
    q2 = run_quiz2(imdb, ckpt_dir, out_dir)
    q3 = run_quiz3(pets, ckpt_dir, out_dir)
    q4 = run_quiz4(flickr, data_dir, ckpt_dir, out_dir)

    print('\n' + '=' * 70 + '\nSUMMARY\n' + '=' * 70)
    for t in (q2['table'], q3['table'], q4['full_table'], q4['subset_table']):
        print(t.to_string(index=False), end='\n\n')
    print(f'Figures and tables saved to: {out_dir}')
    print(f'Total time: {(time.perf_counter() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
