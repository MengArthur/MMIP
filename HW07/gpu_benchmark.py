"""
Measure how fast this GPU runs the HW07 workloads (run it on Colab T4 and on the lab RTX 2080 Ti):

  python gpu_benchmark.py            # ~3 min: matmul, bandwidth, Qwen3-4B FP16 decoding, one DPO step
  python gpu_benchmark.py --quick    # ~10 s: matmul + bandwidth only (no model download)

Results are printed and saved to Output/gpu_benchmark_<gpu>.json.
"""

import argparse
import os
import re
import sys
import time

import torch

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)


def timed(fn, iters):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def matmul_tflops(n=8192, dtype=torch.float16, iters=20):
    a, b = torch.randn(n, n, device='cuda', dtype=dtype), torch.randn(n, n, device='cuda', dtype=dtype)
    timed(lambda: a @ b, 3)
    return 2 * n ** 3 / timed(lambda: a @ b, iters) / 1e12


def bandwidth_gbs(mb=1024, iters=20):
    x = torch.empty(mb * 2**20 // 2, device='cuda', dtype=torch.float16); y = torch.empty_like(x)
    timed(lambda: y.copy_(x), 3)
    return 2 * x.numel() * 2 / timed(lambda: y.copy_(x), iters) / 1e9      # read + write


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--quick', action='store_true'); a = ap.parse_args()
    assert torch.cuda.is_available(), 'needs a CUDA GPU'
    from Code.common import gpu_info, save_json, out_path
    res = gpu_info()
    res['bf16_native'] = torch.cuda.get_device_capability(0) >= (8, 0)
    res['fp16_matmul_tflops'] = round(matmul_tflops(), 1)
    res['fp32_matmul_tflops'] = round(matmul_tflops(4096, torch.float32), 1)
    res['mem_bandwidth_gbs'] = round(bandwidth_gbs(), 1)
    print(res)
    if not a.quick:
        from Code.llm import load_llm, generate_batch, build_chat
        from Code import dpo
        model, tok = load_llm('fp16')
        msg = [[{'role': 'user', 'content': '請詳細介紹台灣的夜市文化，至少 500 字。'}]]
        generate_batch(model, tok, msg, 8)
        for bs in (1, 10):
            t0 = time.perf_counter(); r = generate_batch(model, tok, msg * bs, 128); dt = time.perf_counter() - t0
            res[f'qwen3_4b_fp16_decode_tok_s_batch{bs}'] = round(sum(x['n_new_tokens'] for x in r) / dt, 1)
        res['qwen3_4b_fp16_weights_gib'] = round(torch.cuda.memory_allocated() / 2**30, 2)
        pm = dpo.add_lora(model, r=16)
        enc = dpo.llm_encoder(pm, tok, build_chat)
        pair = {'prompt': '請介紹電腦。', 'chosen': '電腦' * 200, 'rejected': '电脑' * 200}   # ~400 response tokens each
        ctx, scaler_on = dpo._autocast(pm)
        opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=1e-5)
        scaler = torch.amp.GradScaler('cuda', enabled=scaler_on)
        pm.train(); torch.cuda.reset_peak_memory_stats()

        def step():
            c, r, _ = dpo._pair_logps(pm, enc, pair, ctx)
            loss = -torch.nn.functional.logsigmoid(0.1 * (c - r)).mean()
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); opt.zero_grad()
        step()
        res['dpo_lora_step_s (2 x ~420 tokens)'] = round(timed(step, 5), 3)
        res['dpo_peak_memory_gib'] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        print(res)
    name = re.sub(r'[^A-Za-z0-9]+', '_', res['name'])
    save_json(res, out_path(f'gpu_benchmark_{name}.json'))


if __name__ == '__main__':
    main()
