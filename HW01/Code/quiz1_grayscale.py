"""
Quiz 1: RGB to Grayscale Conversion
Basic (15%): Color to Grayscale using OpenCV
Advanced (10%): Custom NumPy implementation, benchmark N=1000 times, analyze differences
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from .utils import benchmark_function, compute_image_metrics


def rgb_to_gray_opencv(bgr_img):
    """
    Basic Task: Convert BGR color image to grayscale using OpenCV cv2.cvtColor.
    Uses OpenCV internal SIMD-optimized implementation of ITU-R BT.601:
    Y = 0.299*R + 0.587*G + 0.114*B
    """
    if bgr_img.ndim == 2:
        return bgr_img.copy()
    if bgr_img.ndim != 3 or bgr_img.shape[2] != 3:
        raise ValueError(f"Input image must be a 3-channel BGR image, got shape {bgr_img.shape}")
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)


def rgb_to_gray_numpy_float(bgr_img):
    """
    Advanced Task: Pure NumPy vectorized implementation with standard floating point weights.
    bgr_img shape: (H, W, 3) where [:, :, 0] is B, [:, :, 1] is G, [:, :, 2] is R.
    """
    if bgr_img.ndim == 2:
        return bgr_img.copy()
    if bgr_img.ndim != 3 or bgr_img.shape[2] != 3:
        raise ValueError(f"Input image must be a 3-channel BGR image, got shape {bgr_img.shape}")
    # Weights for [B, G, R]
    weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
    # Vectorized dot product along the channel dimension
    gray_float = np.dot(bgr_img.astype(np.float32), weights)
    # Round to nearest integer and cast to uint8
    return np.clip(np.round(gray_float), 0, 255).astype(np.uint8)


def rgb_to_gray_numpy_fixedpoint(bgr_img):
    """
    Advanced Task: Exact replica of OpenCV integer fixed-point arithmetic:
    OpenCV uses: Y = (1868 * B + 9617 * G + 4899 * R + 8192) >> 14
    This eliminates floating point rounding discrepancies and achieves bitwise identical results.
    """
    if bgr_img.ndim == 2:
        return bgr_img.copy()
    if bgr_img.ndim != 3 or bgr_img.shape[2] != 3:
        raise ValueError(f"Input image must be a 3-channel BGR image, got shape {bgr_img.shape}")
    b = bgr_img[:, :, 0].astype(np.int32)
    g = bgr_img[:, :, 1].astype(np.int32)
    r = bgr_img[:, :, 2].astype(np.int32)
    gray = (1868 * b + 9617 * g + 4899 * r + 8192) >> 14
    return np.clip(gray, 0, 255).astype(np.uint8)


def run_quiz1(image_path, output_dir, N=1000):
    """
    Execute Quiz 1 pipeline:
    1. Load image
    2. Convert using OpenCV, NumPy float, and NumPy fixed-point
    3. Benchmark N repetitions
    4. Compute quantitative difference metrics
    5. Save visual comparison plot
    """
    os.makedirs(output_dir, exist_ok=True)
    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        raise FileNotFoundError(f"Cannot read image from {image_path}")

    print("\n=======================================================")
    print("           QUIZ 1: COLOR TO GRAYSCALE CONVERSION       ")
    print("=======================================================")
    h, w, c = bgr_img.shape
    print(f"Input Image: {os.path.basename(image_path)} ({w}x{h}, {c} channels)")

    # 1. Benchmarking
    print(f"\n[Benchmarking] Running {N} iterations for each method...")
    t_cv_mean, t_cv_std, gray_cv = benchmark_function(rgb_to_gray_opencv, bgr_img, N=N)
    t_np_mean, t_np_std, gray_np = benchmark_function(rgb_to_gray_numpy_float, bgr_img, N=N)
    t_fp_mean, t_fp_std, gray_fp = benchmark_function(rgb_to_gray_numpy_fixedpoint, bgr_img, N=N)

    print(f"  1. OpenCV (cvtColor):           {t_cv_mean:8.4f} ms ± {t_cv_std:6.4f} ms")
    print(f"  2. NumPy (Vectorized Float):    {t_np_mean:8.4f} ms ± {t_np_std:6.4f} ms  (Speedup vs NP: {t_np_mean/t_cv_mean:.2f}x)")
    print(f"  3. NumPy (Fixed-Point Integer): {t_fp_mean:8.4f} ms ± {t_fp_std:6.4f} ms")

    # 2. Numerical Difference Metrics
    metrics_float = compute_image_metrics(gray_cv, gray_np)
    metrics_fp = compute_image_metrics(gray_cv, gray_fp)

    print(f"\n[Difference Analysis vs OpenCV]")
    print(f"  NumPy Float:")
    print(f"    - Mean Absolute Error (MAE): {metrics_float['mae']:.6f}")
    print(f"    - Mean Squared Error (MSE):  {metrics_float['mse']:.6f}")
    print(f"    - Max Absolute Diff:         {metrics_float['max_diff']}")
    print(f"    - PSNR:                      {metrics_float['psnr']:.2f} dB")

    print(f"  NumPy Fixed-Point (OpenCV Integer Matching):")
    print(f"    - Mean Absolute Error (MAE): {metrics_fp['mae']:.6f}")
    print(f"    - Mean Squared Error (MSE):  {metrics_fp['mse']:.6f}")
    print(f"    - Max Absolute Diff:         {metrics_fp['max_diff']}")
    print(f"    - PSNR:                      {metrics_fp['psnr']:.2f} dB")

    # 3. Visualization Plot (1x3: Original RGB, OpenCV Grayscale, NumPy Grayscale)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    rgb_disp = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    axes[0].imshow(rgb_disp)
    axes[0].set_title("Original Color Image (RGB)", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(gray_cv, cmap='gray')
    axes[1].set_title(f"OpenCV cvtColor ({t_cv_mean:.3f} ms)", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    axes[2].imshow(gray_np, cmap='gray')
    axes[2].set_title(f"NumPy Vectorized Float ({t_np_mean:.3f} ms)", fontsize=12, fontweight='bold')
    axes[2].axis('off')

    plt.suptitle("Quiz 1: Color to Grayscale Conversion (OpenCV vs. Vectorized NumPy)", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_plot_path = os.path.join(output_dir, "quiz1_comparison.png")
    plt.savefig(out_plot_path, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"\n[Saved Output] Plot saved to: {out_plot_path}")

    return {
        't_cv': t_cv_mean,
        't_np': t_np_mean,
        't_fp': t_fp_mean,
        'metrics_float': metrics_float,
        'metrics_fp': metrics_fp
    }
