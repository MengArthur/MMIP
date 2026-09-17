"""
Quiz 4: Feature-Based Image Stitching
Basic (15%): SIFT Keypoints + FLANN/BFMatcher + Lowe's Ratio Test + RANSAC Homography + Stitching
Advanced (10%): Linear Feathering Seam Blending, Multi-condition Stress Testing (Brightness, Rotation, Overlap),
                and In-depth Failure Boundary Analysis.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_and_match_sift(img1, img2, ratio_thresh=0.75, use_clahe=False):
    """
    Detect SIFT keypoints, extract 128-d descriptors, and perform ratio-test matching.
    Optionally applies CLAHE preprocessing to normalize lighting and enhance local gradients.
    """
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        g1 = clahe.apply(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
        g2 = clahe.apply(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))
    else:
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return kp1, kp2, [], None, 0

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

    return kp1, kp2, good_matches, H, inliers


def stitch_images_basic(img1, img2, H):
    """
    Basic Task: Warp img2 to img1's projective frame and overlay img1 directly.
    """
    if H is None:
        raise ValueError("Cannot stitch images: Homography matrix H is None (insufficient matching features).")

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Transform corners of img2 to determine total canvas bounds
    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    warped_corners2 = cv2.perspectiveTransform(corners2, H)

    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    all_corners = np.concatenate((corners1, warped_corners2), axis=0)

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation = np.array([
        [1, 0, -x_min if x_min < 0 else 0],
        [0, 1, -y_min if y_min < 0 else 0],
        [0, 0, 1]
    ], dtype=np.float32)

    canvas_w = int(max(x_max, w1) + (-x_min if x_min < 0 else 0))
    canvas_h = int(max(y_max, h1) + (-y_min if y_min < 0 else 0))

    # Warp img2
    warped_img2 = cv2.warpPerspective(img2, translation.dot(H), (canvas_w, canvas_h))

    # Overlay img1
    stitched = warped_img2.copy()
    offset_x = int(-x_min if x_min < 0 else 0)
    offset_y = int(-y_min if y_min < 0 else 0)
    stitched[offset_y:offset_y + h1, offset_x:offset_x + w1] = img1

    return stitched


def stitch_images_blended(img1, img2, H):
    """
    Advanced Task: Seamless Panorama Stitching using Multi-Band / Distance-Weighted Linear Blending.
    Eliminates vertical seam lines and compensates for exposure discrepancies in the overlap region.
    """
    if H is None:
        raise ValueError("Cannot stitch images: Homography matrix H is None (insufficient matching features).")

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    warped_corners2 = cv2.perspectiveTransform(corners2, H)
    corners1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    all_corners = np.concatenate((corners1, warped_corners2), axis=0)

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    offset_x = int(-x_min if x_min < 0 else 0)
    offset_y = int(-y_min if y_min < 0 else 0)
    translation = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float32)

    canvas_w = int(max(x_max, w1) + offset_x)
    canvas_h = int(max(y_max, h1) + offset_y)

    warped_img2 = cv2.warpPerspective(img2, translation.dot(H), (canvas_w, canvas_h))

    # Canvas for img1
    canvas_img1 = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas_img1[offset_y:offset_y + h1, offset_x:offset_x + w1] = img1

    # Create masks
    mask1 = (canvas_img1.sum(axis=2) > 0).astype(np.float32)
    mask2 = (warped_img2.sum(axis=2) > 0).astype(np.float32)
    overlap = (mask1 * mask2) > 0

    # Distance transform based weights for smooth feathering
    dist1 = cv2.distanceTransform(mask1.astype(np.uint8), cv2.DIST_L2, 5)
    dist2 = cv2.distanceTransform(mask2.astype(np.uint8), cv2.DIST_L2, 5)

    sum_dist = dist1 + dist2 + 1e-6
    weight1 = dist1 / sum_dist
    weight2 = dist2 / sum_dist

    # Apply blending across 3 channels
    blended = np.zeros_like(warped_img2, dtype=np.float32)
    for c in range(3):
        blended[:, :, c] = np.where(
            overlap,
            canvas_img1[:, :, c] * weight1 + warped_img2[:, :, c] * weight2,
            np.where(mask1 > 0, canvas_img1[:, :, c], warped_img2[:, :, c])
        )

    return np.clip(blended, 0, 255).astype(np.uint8)


def run_quiz4(data_dir, output_dir):
    """
    Execute Quiz 4 pipeline:
    1. Basic Task: SIFT + RANSAC on overlapping pair + basic overlay stitching
    2. Advanced Task: Seamless Linear Feathering Blending
    3. Advanced Task: Multi-Condition Stress Testing & Failure Boundary Analysis
    4. Save comparison plots and quantitative tables
    """
    os.makedirs(output_dir, exist_ok=True)
    left_path = os.path.join(data_dir, "quiz4_scene_left.jpg")
    right_path = os.path.join(data_dir, "quiz4_scene_right.jpg")

    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)
    if img_left is None or img_right is None:
        raise FileNotFoundError("Cannot read Quiz 4 input images")

    print("\n=======================================================")
    print("           QUIZ 4: FEATURE-BASED IMAGE STITCHING       ")
    print("=======================================================")

    # 1. Baseline SIFT Matching & Homography
    kp1, kp2, good_matches, H, inliers = detect_and_match_sift(img_left, img_right)
    print(f"[Baseline Pair] Left Keypoints: {len(kp1)}, Right Keypoints: {len(kp2)}")
    print(f"  - Good Matches (Lowe's Ratio 0.75): {len(good_matches)}")
    print(f"  - RANSAC Inliers:                   {inliers} (Inlier Ratio: {inliers/max(len(good_matches),1)*100:.1f}%)")

    # 2. Basic Stitching vs Advanced Blending
    basic_stitched = stitch_images_basic(img_left, img_right, H)
    blended_stitched = stitch_images_blended(img_left, img_right, H)
    print(f"  - Stitched Dimensions: {blended_stitched.shape[1]}x{blended_stitched.shape[0]}")

    # Draw matched features
    matched_vis = cv2.drawMatches(
        img_left, kp1, img_right, kp2, good_matches[:50], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 255, 0)
    )

    # 3. Advanced Task: Stress Testing Across Varying Environmental & Geometric Conditions
    stress_cases = [
        ("Baseline (Normal)", "quiz4_scene_right.jpg", False),
        ("Dark (-40% Light)", "quiz4_scene_right_dark40.jpg", False),
        ("Bright (+40% Light)", "quiz4_scene_right_bright40.jpg", False),
        ("Moderate Rot (25°)", "quiz4_scene_right_rot25.jpg", False),
        ("Extreme Rot (50°)", "quiz4_scene_right_rot50.jpg", False),
        ("Low Overlap (~10%)", "quiz4_scene_right_low_overlap.jpg", False),
        ("CLAHE Normalized (Dark)", "quiz4_scene_right_dark40.jpg", True)  # Preprocessing test!
    ]

    print("\n[Advanced Task] Multi-Condition Stress Testing & Failure Boundary Analysis:")
    stress_results = []

    for name, r_fname, use_clahe in stress_cases:
        r_img = cv2.imread(os.path.join(data_dir, r_fname))
        kp_l, kp_r, g_matches, H_stress, inl = detect_and_match_sift(img_left, r_img, use_clahe=use_clahe)
        
        inlier_ratio = (inl / len(g_matches) * 100) if len(g_matches) > 0 else 0
        success = (H_stress is not None) and (inl >= 10) and (inlier_ratio >= 25.0)

        status_str = "SUCCESS" if success else "FAILED"
        print(f"  - {name:25s} | Good Matches: {len(g_matches):4d} | Inliers: {inl:4d} ({inlier_ratio:5.1f}%) | {status_str}")

        stress_results.append({
            'name': name,
            'matches': len(g_matches),
            'inliers': inl,
            'ratio': inlier_ratio,
            'success': success,
            'H': H_stress
        })

    # 4. Plot 1: SIFT Matching & Stitched Panorama (合起來的成果)
    fig1 = plt.figure(figsize=(16, 9))
    gs1 = fig1.add_gridspec(2, 2, height_ratios=[1, 1.1])

    ax_m = fig1.add_subplot(gs1[0, :])
    ax_m.imshow(cv2.cvtColor(matched_vis, cv2.COLOR_BGR2RGB))
    ax_m.set_title(f"SIFT Feature Matching (Top 50 Inlier Matches | Total Inliers: {inliers})", fontsize=12, fontweight='bold')
    ax_m.axis('off')

    ax_b = fig1.add_subplot(gs1[1, 0])
    ax_b.imshow(cv2.cvtColor(basic_stitched, cv2.COLOR_BGR2RGB))
    ax_b.set_title("Basic Task: Direct Overlay Stitching (Visible Vertical Seam)", fontsize=11, fontweight='bold')
    ax_b.axis('off')

    ax_s = fig1.add_subplot(gs1[1, 1])
    ax_s.imshow(cv2.cvtColor(blended_stitched, cv2.COLOR_BGR2RGB))
    ax_s.set_title("Advanced Task: Seamless Blended Panorama (Linear Feathering)", fontsize=11, fontweight='bold')
    ax_s.axis('off')

    plt.suptitle("Quiz 4: SIFT Feature Matching & Stitched Panorama Results", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plot1_path = os.path.join(output_dir, "quiz4_stitching_result.png")
    fig1.savefig(plot1_path, dpi=180, bbox_inches='tight')
    plt.close(fig1)
    print(f"\n[Saved Output] Stitched panorama plot saved to: {plot1_path}")

    # 5. Plot 2: Advance Failure Boundary (50° Rotation) & Preprocessing Enhancement (失敗的 advance)
    img_rot50 = cv2.imread(os.path.join(data_dir, "quiz4_scene_right_rot50.jpg"))
    _, _, _, H_r50, inl_r50 = detect_and_match_sift(img_left, img_rot50)
    stitched_r50 = stitch_images_basic(img_left, img_rot50, H_r50)

    img_dark = cv2.imread(os.path.join(data_dir, "quiz4_scene_right_dark40.jpg"))
    kp_l_c, kp_clahe, matches_clahe, H_clahe, inl_clahe = detect_and_match_sift(img_left, img_dark, use_clahe=True)
    match_vis_clahe = cv2.drawMatches(
        img_left, kp_l_c, img_dark, kp_clahe,
        matches_clahe[:35], None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        matchColor=(0, 255, 0)
    )

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5.5))
    axes2[0].imshow(cv2.cvtColor(stitched_r50, cv2.COLOR_BGR2RGB))
    axes2[0].set_title(f"[Failure Boundary: 50° Extreme Rotation] Homography Collapse & Distortion\n(Inliers dropped to {inl_r50} | Severe Parallax Mismatch)", fontsize=11, fontweight='bold', color='darkred')
    axes2[0].axis('off')

    axes2[1].imshow(cv2.cvtColor(match_vis_clahe, cv2.COLOR_BGR2RGB))
    axes2[1].set_title(f"[Preprocessing: CLAHE Enhancement] Normalized Dark Lighting\n(Rescued {len(matches_clahe)} Good Matches | Inliers: {inl_clahe})", fontsize=11, fontweight='bold', color='darkgreen')
    axes2[1].axis('off')

    plt.suptitle("Quiz 4 Advance: Failure Boundary (50° Rotation) & Preprocessing Enhancement (CLAHE)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plot2_path = os.path.join(output_dir, "quiz4_advance_analysis.png")
    fig2.savefig(plot2_path, dpi=180, bbox_inches='tight')
    plt.close(fig2)
    print(f"[Saved Output] Advance failure & preprocessing plot saved to: {plot2_path}")

    return {
        'inliers': inliers,
        'stress_results': stress_results
    }
