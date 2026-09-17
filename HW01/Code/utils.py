"""
MMIP HW01 Utilities Module
Contains benchmarking tools, evaluation metrics, and image visualization utilities.
"""

import time
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for terminal execution
import matplotlib.pyplot as plt


def benchmark_function(func, *args, N=1000, warmup=10, **kwargs):
    """
    Benchmark execution time of a callable over N iterations.
    
    Returns:
        mean_ms (float): Mean execution time in milliseconds.
        std_ms (float): Standard deviation of execution time in milliseconds.
        result: The output from the last execution of func.
    """
    # Warmup runs to initialize caches
    for _ in range(warmup):
        _ = func(*args, **kwargs)

    times = []
    result = None
    for _ in range(N):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    mean_ms = float(np.mean(times))
    std_ms = float(np.std(times))
    return mean_ms, std_ms, result


def compute_image_metrics(img1, img2):
    """
    Compute quantitative difference metrics between two images.
    
    Returns:
        dict: {'mae': float, 'mse': float, 'max_diff': float, 'psnr': float}
    """
    diff = np.abs(img1.astype(np.float64) - img2.astype(np.float64))
    mae = float(np.mean(diff))
    mse = float(np.mean(diff ** 2))
    max_diff = float(np.max(diff))
    
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = float(20.0 * np.log10(255.0 / np.sqrt(mse)))

    return {
        'mae': mae,
        'mse': mse,
        'max_diff': max_diff,
        'psnr': psnr
    }


def compute_entropy(gray_img):
    """Calculate Shannon entropy of an 8-bit grayscale image."""
    hist, _ = np.histogram(gray_img.ravel(), bins=256, range=(0, 256))
    prob = hist / float(hist.sum())
    prob = prob[prob > 0]
    return float(-np.sum(prob * np.log2(prob)))


def compute_intensity_stats(gray_img):
    """Return min, max, mean, std of pixel intensity."""
    return {
        'min': int(np.min(gray_img)),
        'max': int(np.max(gray_img)),
        'mean': float(np.mean(gray_img)),
        'std': float(np.std(gray_img))
    }
