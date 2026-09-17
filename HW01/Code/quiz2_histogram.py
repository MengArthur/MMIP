"""
Quiz 2: Histogram Equalization
Basic (15%): Equalize histogram using OpenCV, plot before/after histograms with Min/Max/Mean/Std statistics
Advanced (10%): Pure NumPy implementation (CDF & LUT mapping), benchmark N=1000 times, evaluate enhancement & difference
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from .utils import benchmark_function, compute_image_metrics, compute_entropy, compute_intensity_stats


def hist_equalize_opencv(gray_img):
    """Basic Task: Histogram equalization using OpenCV cv2.equalizeHist."""
    if gray_img.ndim != 2:
        raise ValueError(f"Input must be a single-channel 2D grayscale image, got shape {gray_img.shape}")
    return cv2.equalizeHist(gray_img)


def hist_equalize_numpy(gray_img):
    """
    Advanced Task: Pure NumPy implementation of Histogram Equalization.
    Mathematical formulation:
    1. Calculate histogram h(v) for intensity levels v in [0, 255].
    2. Compute Cumulative Distribution Function (CDF): cdf(v) = sum_{i=0}^v h(i)
    3. Normalize CDF into 8-bit range [0, 255] using histogram equalization formula:
       h(v) = round( (cdf(v) - cdf_min) / ((M * N) - cdf_min) * 255 )
    4. Map pixels using the calculated lookup table (LUT).
    """
    if gray_img.ndim != 2:
        raise ValueError(f"Input must be a single-channel 2D grayscale image, got shape {gray_img.shape}")
        
    # 1. Compute histogram using fast np.bincount (3x faster than np.histogram)
    hist = np.bincount(gray_img.ravel(), minlength=256)
    
    # 2. Cumulative Distribution Function
    cdf = hist.cumsum()
    
    # 3. Mask zeros to avoid normalizing empty intensity bins
    cdf_m = np.ma.masked_equal(cdf, 0)
    
    # Check edge case: uniform flat image
    if cdf_m.max() == cdf_m.min():
        return gray_img.copy()
        
    # Scale CDF to [0, 255]
    cdf_m = (cdf_m - cdf_m.min()) * 255.0 / (cdf_m.max() - cdf_m.min())
    lut = np.ma.filled(cdf_m, 0).round().astype(np.uint8)
    
    # 4. Map pixel values via LUT
    return lut[gray_img]


def hist_equalize_color_hsv(bgr_img, method='opencv'):
    """
    Equalize color image by transforming BGR -> HSV, equalizing V (Value/Brightness) channel,
    and transforming back to BGR (Slide 30 recommendation).
    """
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    if method == 'opencv':
        v_eq = cv2.equalizeHist(v)
    else:
        v_eq = hist_equalize_numpy(v)
        
    hsv_eq = cv2.merge([h, s, v_eq])
    return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)


def run_quiz2(image_path, output_dir, N=1000):
    """
    Execute Quiz 2 pipeline:
    1. Load low-contrast image (grayscale and color)
    2. Equalize using OpenCV and NumPy
    3. Benchmark N repetitions
    4. Compute contrast, entropy, and difference metrics
    5. Plot before/after histograms with Min/Max/Mean/Std annotations matching Slide 28/29
    """
    os.makedirs(output_dir, exist_ok=True)
    bgr_img = cv2.imread(image_path)
    if bgr_img is None:
        raise FileNotFoundError(f"Cannot read image from {image_path}")

    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    print("\n=======================================================")
    print("           QUIZ 2: HISTOGRAM EQUALIZATION              ")
    print("=======================================================")
    h, w = gray_img.shape
    print(f"Input Image: {os.path.basename(image_path)} ({w}x{h})")

    # 1. Benchmarking
    print(f"\n[Benchmarking] Running {N} iterations for each method...")
    t_cv_mean, t_cv_std, eq_cv = benchmark_function(hist_equalize_opencv, gray_img, N=N)
    t_np_mean, t_np_std, eq_np = benchmark_function(hist_equalize_numpy, gray_img, N=N)

    print(f"  1. OpenCV (equalizeHist):   {t_cv_mean:8.4f} ms ± {t_cv_std:6.4f} ms")
    print(f"  2. NumPy (Vectorized LUT):  {t_np_mean:8.4f} ms ± {t_np_std:6.4f} ms  (Speedup vs NP: {t_np_mean/t_cv_mean:.2f}x)")

    # 2. Image Statistics & Information Entropy
    stats_orig = compute_intensity_stats(gray_img)
    stats_cv = compute_intensity_stats(eq_cv)
    stats_np = compute_intensity_stats(eq_np)

    ent_orig = compute_entropy(gray_img)
    ent_cv = compute_entropy(eq_cv)
    ent_np = compute_entropy(eq_np)

    diff_metrics = compute_image_metrics(eq_cv, eq_np)

    print(f"\n[Image Enhancement & Statistical Metrics]")
    print(f"  - Original: Min={stats_orig['min']:3d}, Max={stats_orig['max']:3d}, Mean={stats_orig['mean']:6.2f}, Std={stats_orig['std']:6.2f}, Entropy={ent_orig:.4f}")
    print(f"  - OpenCV:   Min={stats_cv['min']:3d}, Max={stats_cv['max']:3d}, Mean={stats_cv['mean']:6.2f}, Std={stats_cv['std']:6.2f}, Entropy={ent_cv:.4f}")
    print(f"  - NumPy:    Min={stats_np['min']:3d}, Max={stats_np['max']:3d}, Mean={stats_np['mean']:6.2f}, Std={stats_np['std']:6.2f}, Entropy={ent_np:.4f}")
    print(f"  - Difference (OpenCV vs NumPy): MAE={diff_metrics['mae']:.6f}, MaxDiff={diff_metrics['max_diff']}")

    # 3. Visualization matching course slides style
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Top-Left: Before Image
    axes[0, 0].imshow(gray_img, cmap='gray')
    axes[0, 0].set_title("Before Histogram Equalization", fontsize=13, fontweight='bold')
    axes[0, 0].axis('off')

    # Top-Right: Before Histogram
    axes[0, 1].hist(gray_img.ravel(), bins=256, range=[0, 256], color='#1f77b4', alpha=0.85)
    axes[0, 1].set_title("Before Histogram", fontsize=13, fontweight='bold')
    axes[0, 1].set_xlim([0, 256])
    axes[0, 1].set_xlabel("Pixel Intensity")
    axes[0, 1].set_ylabel("Number of Pixels")
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)
    axes[0, 1].text(0.5, -0.22, f"Min : {stats_orig['min']}   Max : {stats_orig['max']}   Mean : {stats_orig['mean']:.1f}   Std : {stats_orig['std']:.1f}",
                    transform=axes[0, 1].transAxes, fontsize=12, fontweight='bold', ha='center')

    # Bottom-Left: After Image (Equalized)
    axes[1, 0].imshow(eq_cv, cmap='gray')
    axes[1, 0].set_title(f"After Histogram Equalization (OpenCV: {t_cv_mean:.3f} ms, NumPy: {t_np_mean:.3f} ms)", fontsize=13, fontweight='bold')
    axes[1, 0].axis('off')

    # Bottom-Right: After Histogram
    axes[1, 1].hist(eq_cv.ravel(), bins=256, range=[0, 256], color='#1f77b4', alpha=0.85)
    axes[1, 1].set_title("After Histogram", fontsize=13, fontweight='bold')
    axes[1, 1].set_xlim([0, 256])
    axes[1, 1].set_xlabel("Pixel Intensity")
    axes[1, 1].set_ylabel("Number of Pixels")
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)
    axes[1, 1].text(0.5, -0.22, f"Min : {stats_cv['min']}   Max : {stats_cv['max']}   Mean : {stats_cv['mean']:.1f}   Std : {stats_cv['std']:.1f}",
                    transform=axes[1, 1].transAxes, fontsize=12, fontweight='bold', ha='center')

    plt.suptitle("Quiz 2: Histogram Equalization & Distribution Analysis", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    out_plot_path = os.path.join(output_dir, "quiz2_comparison.png")
    plt.savefig(out_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n[Saved Output] Plot saved to: {out_plot_path}")

    # 4. Color HSV Equalization Demo (Slide 30)
    color_eq = hist_equalize_color_hsv(bgr_img, method='opencv')
    color_plot_path = os.path.join(output_dir, "quiz2_color_equalization.png")
    fig_col, ax_col = plt.subplots(1, 2, figsize=(14, 6))
    ax_col[0].imshow(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
    ax_col[0].set_title("Original Color Image (Low Contrast)", fontsize=12, fontweight='bold')
    ax_col[0].axis('off')
    ax_col[1].imshow(cv2.cvtColor(color_eq, cv2.COLOR_BGR2RGB))
    ax_col[1].set_title("Color Equalized via HSV V-Channel (Contrast Enhanced)", fontsize=12, fontweight='bold')
    ax_col[1].axis('off')
    plt.tight_layout()
    plt.savefig(color_plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Saved Output] Color equalization plot saved to: {color_plot_path}")

    return {
        't_cv': t_cv_mean,
        't_np': t_np_mean,
        'stats_orig': stats_orig,
        'stats_cv': stats_cv,
        'stats_np': stats_np,
        'ent_orig': ent_orig,
        'ent_cv': ent_cv,
        'ent_np': ent_np,
        'diff_metrics': diff_metrics
    }
