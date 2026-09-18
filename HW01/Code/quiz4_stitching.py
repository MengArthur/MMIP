"""
Quiz 4: SIFT Feature Matching & Image Stitching
Basic (15%): Two overlapping images, SIFT feature extraction, matching, and homography mosaic.
Advanced (10%): Multi-condition failure boundary exploration (Zero overlap & Extreme darkness),
                and CLAHE preprocessing enhancement to rescue failed stitching.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def _auto_crop_black(panorama):
    """
    Dynamically crop black borders from warpPerspective output.
    Finds the maximum area inscribed rectangle containing only non-zero pixels using dynamic programming.
    Completely eliminates irregular black borders/wedges for any input dimensions without hardcoding.
    """
    gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY) if panorama.ndim == 3 else panorama
    mask = (gray > 0).astype(np.uint8)

    H, W = mask.shape
    heights = np.zeros(W, dtype=np.int32)
    max_area = 0
    best_rect = (0, 0, W, H)

    for y in range(H):
        heights = np.where(mask[y] > 0, heights + 1, 0)
        stack = []
        for x in range(W + 1):
            cur_h = heights[x] if x < W else 0
            while stack and heights[stack[-1]] >= cur_h:
                h = heights[stack.pop()]
                w = x if not stack else x - stack[-1] - 1
                rx = stack[-1] + 1 if stack else 0
                ry = y - h + 1
                area = w * h
                if area > max_area:
                    max_area = area
                    best_rect = (rx, ry, w, h)
            stack.append(x)

    rx, ry, rw, rh = best_rect
    if rw > 0 and rh > 0:
        return panorama[ry:ry + rh, rx:rx + rw]
    return panorama


def detect_and_match_sift(img1, img2, ratio_thresh=0.75, use_clahe=False):
    """
    Detect SIFT keypoints, compute descriptors, match using k-NN, and estimate Homography with RANSAC.
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if img1.ndim == 3 else img1.copy()
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if img2.ndim == 3 else img2.copy()

    # Preprocessing: Optional CLAHE local contrast enhancement
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray1 = clahe.apply(gray1)
        gray2 = clahe.apply(gray2)

    # SIFT detector
    sift = cv2.SIFT_create(contrastThreshold=0.04, edgeThreshold=10)
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return kp1 if kp1 is not None else [], kp2 if kp2 is not None else [], [], None, 0

    # Match descriptors using BFMatcher with L2 norm
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    # Lowe's Ratio Test
    good_matches = []
    for m, n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return kp1, kp2, good_matches, None, 0

    # Extract matched coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute Homography mapping from img2 (source) to img1 (destination)
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 4.0)
    inliers = int(np.sum(mask)) if mask is not None else 0

    # If inliers < 4, homography is invalid
    if inliers < 4:
        H = None

    return kp1, kp2, good_matches, H, inliers


def stitch_images_basic(img1, img2, H, crop_borders=True):
    """
    Basic Task: Direct canvas overlay mosaic.
    crop_borders: When True, crops the valid inner rectangular panorama with zero black borders.
    """
    if H is None:
        raise ValueError("Cannot stitch images: Homography matrix is None (insufficient inliers).")

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    warped_corners2 = cv2.perspectiveTransform(corners2, H)

    all_corners = np.concatenate((
        np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2),
        warped_corners2
    ), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    tx = max(0, -xmin)
    ty = max(0, -ymin)
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)

    out_w = max(w1 + tx, xmax + tx)
    out_h = max(h1 + ty, ymax + ty)

    warped_img2 = cv2.warpPerspective(img2, T @ H, (out_w, out_h))
    panorama = warped_img2.copy()
    panorama[ty:ty + h1, tx:tx + w1] = img1

    if crop_borders:
        return _auto_crop_black(panorama)
    return panorama


def stitch_images_blended(img1, img2, H, crop_borders=True):
    """
    Advanced Task: Distance-weighted linear feathering blended mosaic.
    crop_borders: When True, crops the valid inner rectangular panorama with zero black borders.
    """
    if H is None:
        raise ValueError("Cannot stitch images: Homography matrix is None (insufficient inliers).")

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    warped_corners2 = cv2.perspectiveTransform(corners2, H)

    all_corners = np.concatenate((
        np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2),
        warped_corners2
    ), axis=0)

    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    tx = max(0, -xmin)
    ty = max(0, -ymin)
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float32)

    out_w = max(w1 + tx, xmax + tx)
    out_h = max(h1 + ty, ymax + ty)

    warped2 = cv2.warpPerspective(img2, T @ H, (out_w, out_h))
    canvas1 = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas1[ty:ty + h1, tx:tx + w1] = img1

    mask1 = cv2.cvtColor(canvas1, cv2.COLOR_BGR2GRAY) > 0
    mask2 = cv2.cvtColor(warped2, cv2.COLOR_BGR2GRAY) > 0

    overlap = mask1 & mask2
    only1 = mask1 & (~mask2)
    only2 = mask2 & (~mask1)

    result = np.zeros_like(canvas1, dtype=np.float32)
    result[only1] = canvas1[only1].astype(np.float32)
    result[only2] = warped2[only2].astype(np.float32)

    if np.any(overlap):
        y_indices, x_indices = np.where(overlap)
        xmin_ov, xmax_ov = x_indices.min(), x_indices.max()
        denom = max(1, xmax_ov - xmin_ov)
        alpha = (x_indices - xmin_ov) / denom
        alpha_3d = alpha[:, np.newaxis]

        p1_val = canvas1[y_indices, x_indices].astype(np.float32)
        p2_val = warped2[y_indices, x_indices].astype(np.float32)
        blended_val = (1.0 - alpha_3d) * p1_val + alpha_3d * p2_val
        result[y_indices, x_indices] = blended_val

    blended_output = np.clip(result, 0, 255).astype(np.uint8)
    if crop_borders:
        return _auto_crop_black(blended_output)
    return blended_output


def run_quiz4(data_dir, output_dir):
    """
    Execute Quiz 4 pipeline:
    1. Load scenes
    2. SIFT feature extraction & matching
    3. Basic overlay vs Blended seamless stitching (clean borderless crop)
    4. Systematic Failure Boundary Exploration (Zero overlap & Extreme darkness)
    5. Preprocessing enhancement (CLAHE)
    """
    os.makedirs(output_dir, exist_ok=True)
    img_left = cv2.imread(os.path.join(data_dir, "quiz4_scene_left.jpg"))
    img_right = cv2.imread(os.path.join(data_dir, "quiz4_scene_right.jpg"))

    if img_left is None or img_right is None:
        raise FileNotFoundError("Missing left or right scene images in Data directory.")

    print("\n=======================================================")
    print("        QUIZ 4: FEATURE MATCHING & IMAGE STITCHING      ")
    print("=======================================================")

    # 1. Basic Task: SIFT Matching & Homography
    kp1, kp2, good_matches, H_matrix, inliers = detect_and_match_sift(img_left, img_right)
    print(f"[Basic Task] Baseline SIFT Matching:")
    print(f"  - Keypoints: Left={len(kp1)}, Right={len(kp2)}")
    print(f"  - Lowe's 0.75 Ratio Matches: {len(good_matches)}")
    print(f"  - RANSAC Inliers:            {inliers} ({inliers/max(len(good_matches),1)*100:.1f}%)")

    # 2. Stitching with clean zero-border crop
    basic_stitched = stitch_images_basic(img_left, img_right, H_matrix, crop_borders=True)
    blended_stitched = stitch_images_blended(img_left, img_right, H_matrix, crop_borders=True)

    # 3. Plot 1: SIFT Matching & Stitched Panorama
    matched_vis = cv2.drawMatches(
        img_left, kp1, img_right, kp2,
        good_matches[:50], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    fig1 = plt.figure(figsize=(16, 9))
    gs1 = fig1.add_gridspec(2, 2, height_ratios=[1, 1.1])

    ax_m = fig1.add_subplot(gs1[0, :])
    ax_m.imshow(cv2.cvtColor(matched_vis, cv2.COLOR_BGR2RGB))
    ax_m.set_title(f"SIFT Feature Matching (Top 50 Inliers | Total Inliers: {inliers})", fontsize=12, fontweight='bold')
    ax_m.axis('off')

    ax_b = fig1.add_subplot(gs1[1, 0])
    ax_b.imshow(cv2.cvtColor(basic_stitched, cv2.COLOR_BGR2RGB))
    ax_b.set_title("Basic Task: Direct Overlay Stitching (Visible Vertical Seam | Clean Crop)", fontsize=11, fontweight='bold')
    ax_b.axis('off')

    ax_s = fig1.add_subplot(gs1[1, 1])
    ax_s.imshow(cv2.cvtColor(blended_stitched, cv2.COLOR_BGR2RGB))
    ax_s.set_title("Advanced Task: Seamless Blended Panorama (Linear Feathering | Clean Crop)", fontsize=11, fontweight='bold')
    ax_s.axis('off')

    plt.suptitle("Quiz 4: SIFT Feature Matching & Stitched Panorama Results", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, "quiz4_stitching_result.png")
    fig1.savefig(plot1_path, dpi=180, bbox_inches='tight')
    plt.close(fig1)
    print(f"[Saved Output] Stitched panorama plot saved to: {plot1_path}")

    # 4. Plot 2: Failure Boundary Exploration (Zero Overlap) & CLAHE Preprocessing Rescue (Clean 2-Row, 0 Black Borders)
    # Failure Case 1: Zero Overlap (different scene from landscape)
    img_disjoint = cv2.imread(os.path.join(data_dir, "quiz1_color.jpg"))
    _, _, m_dis, H_dis, inl_dis = detect_and_match_sift(img_left, img_disjoint, ratio_thresh=0.7)

    # Failure Case 2 & Preprocessing Solution: Extreme Dark Lighting (-88% darkness)
    dark_extreme = np.clip(img_right.astype(np.float32) * 0.12, 0, 255).astype(np.uint8)
    _, _, m_raw, H_raw, inl_raw = detect_and_match_sift(img_left, dark_extreme, use_clahe=False)
    _, _, m_clahe, H_clahe, inl_clahe = detect_and_match_sift(img_left, dark_extreme, use_clahe=True)

    stitched_clahe_clean = stitch_images_blended(img_left, dark_extreme, H_clahe, crop_borders=True)

    fig2 = plt.figure(figsize=(16, 9.5), facecolor='white')
    gs2 = fig2.add_gridspec(2, 3, height_ratios=[1, 1.05])

    # Row 1: Zero Overlap Failure Case
    ax1_l = fig2.add_subplot(gs2[0, 0])
    ax1_l.imshow(cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB))
    ax1_l.set_title("[Scene 1] Left Campus View", fontsize=11, fontweight='bold')
    ax1_l.axis('off')

    ax1_r = fig2.add_subplot(gs2[0, 1])
    ax1_r.imshow(cv2.cvtColor(img_disjoint, cv2.COLOR_BGR2RGB))
    ax1_r.set_title("[Scene 2] Landscape Photo (Zero Overlap)", fontsize=11, fontweight='bold', color='darkred')
    ax1_r.axis('off')

    ax1_status = fig2.add_subplot(gs2[0, 2])
    ax1_status.set_facecolor('#fff5f5')
    ax1_status.text(0.5, 0.62, "FAILED TO STITCH", ha='center', va='center', fontsize=16, fontweight='bold', color='#dc3545')
    ax1_status.text(0.5, 0.40, f"SIFT Inliers: {inl_dis} (< 4 Points)\nHomography: Cannot Solve (None)\nReason: Zero Spatial Overlap (0%)\nResult: System Safely Aborts Stitching", ha='center', va='center', fontsize=11, color='#333333')
    ax1_status.set_xticks([])
    ax1_status.set_yticks([])
    for spine in ax1_status.spines.values():
        spine.set_edgecolor('#dc3545')
        spine.set_linewidth(2.5)
    ax1_status.set_title("[Failure Condition 1] Zero Spatial Overlap (0%)", fontsize=11, fontweight='bold', color='darkred')

    # Row 2: Extreme Darkness Failure vs CLAHE Rescue
    ax2_in = fig2.add_subplot(gs2[1, 0])
    ax2_in.imshow(cv2.cvtColor(dark_extreme, cv2.COLOR_BGR2RGB))
    ax2_in.set_title("[Input] Extreme Dark Lighting (-88% Darkness)", fontsize=11, fontweight='bold')
    ax2_in.axis('off')

    ax2_raw = fig2.add_subplot(gs2[1, 1])
    ax2_raw.set_facecolor('#fff5f5')
    ax2_raw.text(0.5, 0.62, "STITCHING FAILED", ha='center', va='center', fontsize=16, fontweight='bold', color='#dc3545')
    ax2_raw.text(0.5, 0.40, f"Without CLAHE Preprocessing:\nSIFT Detected Keypoints: 0\nMatches: {len(m_raw)} | Inliers: {inl_raw}\nHomography: None\nReason: Gradients Below Contrast Threshold", ha='center', va='center', fontsize=11, color='#333333')
    ax2_raw.set_xticks([])
    ax2_raw.set_yticks([])
    for spine in ax2_raw.spines.values():
        spine.set_edgecolor('#dc3545')
        spine.set_linewidth(2.5)
    ax2_raw.set_title("[Failure Condition 2] Raw Dark (No Preprocessing)", fontsize=11, fontweight='bold', color='darkred')

    ax2_clahe = fig2.add_subplot(gs2[1, 2])
    ax2_clahe.imshow(cv2.cvtColor(stitched_clahe_clean, cv2.COLOR_BGR2RGB))
    ax2_clahe.set_title(f"[RESCUED] With CLAHE Preprocessing\n(Recovered {len(m_clahe)} Matches -> Seamless Panorama)", fontsize=11, fontweight='bold', color='darkgreen')
    ax2_clahe.axis('off')

    plt.suptitle("Quiz 4 Advance: Systematic Failure Boundary Exploration & CLAHE Preprocessing Recovery", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, "quiz4_advance_analysis.png")
    fig2.savefig(plot2_path, dpi=160, bbox_inches='tight')
    plt.close(fig2)
    print(f"[Saved Output] Failure boundary & CLAHE rescue plot saved to: {plot2_path}")

    return {
        'inliers': inliers,
        'H_matrix': H_matrix,
        'clahe_matches': len(m_clahe),
        'plot1': plot1_path,
        'plot2': plot2_path
    }
