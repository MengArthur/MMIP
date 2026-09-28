"""
MMIP HW03: Convolutional Neural Networks -- one-command pipeline.

Runs Quiz 1-3 end to end. Trained models are cached in Checkpoints/ (see
Code/train.py); the first run trains all 10 experiments (~1-1.5 h on a Colab
T4 GPU, many hours on CPU),
later runs reuse the cached checkpoints and only regenerate evaluation/figures.
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

from Code.data import load_cifar10
from Code.quiz1_dataset import run_quiz1
from Code.quiz2_cnn import run_quiz2
from Code.quiz3_generalization import run_quiz3


def main():
    t0 = time.perf_counter()
    data_dir = os.path.join(BASE_DIR, 'Data')
    ckpt_dir = os.path.join(BASE_DIR, 'Checkpoints')
    output_dir = os.path.join(BASE_DIR, 'Output')
    os.makedirs(output_dir, exist_ok=True)
    from Code.train import DEVICE
    dev_str = f"{DEVICE}" + (f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == 'cuda' else "")
    print(f"PyTorch {torch.__version__} | device: {dev_str} | threads: {torch.get_num_threads()}")

    data = load_cifar10(data_dir)
    run_quiz1(data, output_dir)
    q2 = run_quiz2(data, ckpt_dir, output_dir)
    q3 = run_quiz3(data, ckpt_dir, output_dir)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(q2['comparison'][['model', 'total_params', 'test_top1', 'test_top5', 'test_macro_auc']].to_string(index=False))
    print(q3['aug_table'][['experiment', 'best_val_acc', 'test_top1', 'train_val_acc_gap']].to_string(index=False))
    print(f"\nFigures and CSV tables saved to: {output_dir}")
    print(f"Total time: {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == '__main__':
    main()
