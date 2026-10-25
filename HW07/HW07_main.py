"""
MMIP HW07: multimodal language models -- one-command pipeline.

  python HW07_main.py                      # all parts (about 1.5 h on a Colab T4)
  python HW07_main.py --parts A B          # choose parts: A, B, C (LLM DPO), CV (VLM DPO)
  python HW07_main.py --no-gemini          # skip the Gemini judge (rule-based metrics only)

Every generation run, Gemini answer and LoRA adapter is cached in Output/ and Checkpoints/, so an
interrupted run resumes where it stopped. Needs a CUDA GPU (T4 16 GB or RTX 2080 Ti 11 GB).
"""

import argparse
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

import torch  # noqa: E402

from Code.common import gpu_info  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', nargs='+', default=['A', 'B', 'C', 'CV'], choices=['A', 'B', 'C', 'CV'])
    ap.add_argument('--no-gemini', action='store_true')
    a = ap.parse_args()
    t0 = time.perf_counter()
    print(f'PyTorch {torch.__version__} | {gpu_info()}')
    tables = []
    if 'A' in a.parts:
        from Code.partA_llm import run_partA
        r = run_partA(use_gemini=not a.no_gemini)
        tables += [('Part A prompting techniques (FP16)', r['techniques']), ('Part A precision x prompt', r['quant']),
                   ('Part A batch-1 memory profile', r['profile'])]
    if 'B' in a.parts:
        from Code.partB_vlm import run_partB
        r = run_partB(use_gemini=not a.no_gemini)
        tables += [('Part B VLM comparison', r['summary'])]
    if 'C' in a.parts:
        from Code.partC_dpo_llm import run_partC_llm
        r = run_partC_llm()
        tables += [('Part C aspect (40 held-out prompts)', r['aspect']), ('Part C general ability (100 questions)', r['general'])]
    if 'CV' in a.parts:
        from Code.partC_dpo_vlm import run_partC_vlm
        r = run_partC_vlm()
        tables += [('Part C advanced POPE', r['pope']), ('Part C advanced CHAIR', r['chair'])]
    print('\n' + '=' * 80 + '\nSUMMARY\n' + '=' * 80)
    for title, t in tables:
        print(f'\n## {title}\n{t.to_string(index=False)}')
    print(f'\nTotal time: {(time.perf_counter() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
