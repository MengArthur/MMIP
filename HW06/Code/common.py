"""Shared utilities: device, seeding, JSON helpers, GPU-memory sampling, OOM-adaptive batching."""

import json
import os
import random
import sys
import threading
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')   # less fragmentation on 11 GB cards

import numpy as np
import torch
import matplotlib
if 'ipykernel' not in sys.modules:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: F401  (re-exported for the part modules)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR, CKPT_DIR, OUT_DIR = (os.path.join(BASE, d) for d in ('Data', 'Checkpoints', 'Output'))


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def out_path(name):
    os.makedirs(OUT_DIR, exist_ok=True)
    return os.path.join(OUT_DIR, name)


def ckpt_path(name):
    os.makedirs(CKPT_DIR, exist_ok=True)
    return os.path.join(CKPT_DIR, name)


def gpu_info():
    if DEVICE.type != 'cuda':
        return {'device': 'cpu'}
    p = torch.cuda.get_device_properties(0)
    return {'device': 'cuda', 'name': p.name, 'vram_gb': round(p.total_memory / 2**30, 2),
            'capability': f'{p.major}.{p.minor}', 'torch': torch.__version__}


def default_dtype():
    """fp16 on every CUDA GPU (T4 / 2080 Ti have no native bf16); fp32 on CPU (local smoke tests)."""
    return torch.float16 if DEVICE.type == 'cuda' else torch.float32


def free_gpu():
    import gc
    gc.collect()
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()


def logits_finite(model, inputs):
    """One forward pass; False if any logit is inf/nan (fp16 overflow -> reload in bf16)."""
    with torch.no_grad():
        out = model(**inputs)
    return bool(torch.isfinite(out.logits.float()).all())


# ----------------------------------------------------------------------------- GPU memory
class _NVML:
    """Memory used by *this process* on GPU 0 (includes the CUDA context). None if unavailable,
    e.g. pynvml missing or the PID is hidden by a container."""

    def __init__(self):
        self.handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nv = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            if self.read() is None:
                self.handle = None
        except Exception:
            self.handle = None

    def read(self):
        if self.handle is None:
            return None
        try:
            for p in self.nv.nvmlDeviceGetComputeRunningProcesses(self.handle):
                if p.pid == os.getpid() and p.usedGpuMemory:
                    return p.usedGpuMemory
        except Exception:
            pass
        return None


class GPUMemorySampler:
    """
    Samples GPU memory every `interval` seconds in a background thread while the `with` block runs:
      allocated : memory held by PyTorch tensors (weights + KV cache + activations)
      reserved  : allocated + PyTorch caching-allocator pool
      process   : what nvidia-smi shows for this process (reserved + CUDA context), via NVML
    Reports average and peak (MiB). Peak allocated also uses torch.cuda.max_memory_allocated(),
    which catches spikes shorter than the sampling interval.
    """

    def __init__(self, interval=0.02):
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._nvml = _NVML() if DEVICE.type == 'cuda' else None

    def _run(self):
        while not self._stop.is_set():
            self.samples.append((torch.cuda.memory_allocated(), torch.cuda.memory_reserved(),
                                 self._nvml.read() if self._nvml else None))
            time.sleep(self.interval)

    def __enter__(self):
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
        if DEVICE.type == 'cuda':
            torch.cuda.synchronize()
            self._stop.set()
            self._thread.join()
            self.peak_allocated = torch.cuda.max_memory_allocated()
        return False

    def summary(self):
        mib = 2 ** 20
        if DEVICE.type != 'cuda' or not self.samples:
            return {'avg_allocated_mib': None, 'peak_allocated_mib': None, 'avg_reserved_mib': None,
                    'avg_process_mib': None, 'n_samples': 0, 'seconds': round(self.seconds, 2)}
        a = np.array([s[0] for s in self.samples], dtype=float)
        r = np.array([s[1] for s in self.samples], dtype=float)
        p = [s[2] for s in self.samples if s[2] is not None]
        return {'avg_allocated_mib': round(a.mean() / mib, 1), 'peak_allocated_mib': round(self.peak_allocated / mib, 1),
                'avg_reserved_mib': round(r.mean() / mib, 1), 'peak_reserved_mib': round(r.max() / mib, 1),
                'avg_process_mib': round(float(np.mean(p)) / mib, 1) if p else None,
                'n_samples': len(self.samples), 'seconds': round(self.seconds, 2)}


# ----------------------------------------------------------------------------- batching
def run_batched(fn, items, batch_size, min_batch=1):
    """Apply fn(list) -> list over items in chunks; on CUDA OOM halve the chunk size and retry.
    Returns (results, batch size that finally worked)."""
    out, i, bs = [], 0, batch_size
    while i < len(items):
        chunk = items[i:i + bs]
        try:
            out.extend(fn(chunk))
            i += len(chunk)
        except torch.cuda.OutOfMemoryError:
            free_gpu()
            if bs <= min_batch:
                raise
            bs = max(min_batch, bs // 2)
            print(f'  [OOM] retrying with batch size {bs}')
    return out, bs
