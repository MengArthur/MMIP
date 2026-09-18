"""
Data Generator for MMIP HW01
Generates high-quality synthetic and realistic test images for all 4 Quizzes.
"""

import os
import cv2
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(DATA_DIR, exist_ok=True)


def create_quiz1_color_image():
    """Generate a rich colorful image with gradients, shapes, and textures for RGB-to-gray comparison."""
    h, w = 600, 800
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Sky gradient (sunset transition: deep blue to orange/red)
    for y in range(h // 2):
        ratio = y / (h // 2)
        r = int(255 * ratio + 30 * (1 - ratio))
        g = int(100 * ratio + 50 * (1 - ratio))
        b = int(40 * ratio + 200 * (1 - ratio))
        img[y, :] = (b, g, r)  # BGR

    # Sea gradient with specular highlights
    for y in range(h // 2, h):
        ratio = (y - h // 2) / (h // 2)
        b = int(140 * (1 - ratio) + 40 * ratio)
        g = int(80 * (1 - ratio) + 30 * ratio)
        r = int(20 * (1 - ratio) + 10 * ratio)
        img[y, :] = (b, g, r)

    # Sun
    cv2.circle(img, (w // 2, h // 2 - 50), 60, (50, 220, 255), -1)

    # Mountains / Hills
    pts = np.array([[0, h // 2], [150, h // 2 - 80], [350, h // 2 - 40], [500, h // 2 - 110], [700, h // 2 - 60], [w, h // 2]], np.int32)
    cv2.fillPoly(img, [pts], (40, 45, 60))

    # A stylized boat
    boat_pts = np.array([[220, h // 2 + 100], [340, h // 2 + 100], [320, h // 2 + 140], [240, h // 2 + 140]], np.int32)
    cv2.fillPoly(img, [boat_pts], (25, 25, 180))  # Crimson hull
    cv2.line(img, (280, h // 2 + 100), (280, h // 2 + 30), (200, 200, 200), 4)  # Mast
    sail_pts = np.array([[282, h // 2 + 35], [330, h // 2 + 85], [282, h // 2 + 85]], np.int32)
    cv2.fillPoly(img, [sail_pts], (230, 240, 240))  # Sail

    # Text label
    cv2.putText(img, "MMIP HW01 - Benchmark Scene", (30, 560), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    path = os.path.join(DATA_DIR, "quiz1_color.jpg")
    cv2.imwrite(path, img)
    print(f"Generated: {path}")


def create_quiz2_low_contrast_image():
    """Generate a low-contrast, underexposed night image requiring histogram equalization."""
    h, w = 600, 800
    # Base background: dark murky gradient
    img = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        val = int(25 + 35 * (y / h))
        img[y, :] = val

    # Add dark objects with subtle intensity variations (intensities between 20 and 80)
    # Harbor dock
    cv2.rectangle(img, (50, 350), (750, 580), 55, -1)
    cv2.rectangle(img, (70, 370), (400, 550), 45, -1)

    # Distant buildings with dim lights
    for i in range(12):
        bx = 80 + i * 55
        by = 220 + (i % 3) * 20
        bw = 40
        bh = 350 - by
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), 40 + (i % 4) * 8, -1)
        # Dim windows
        for wy in range(by + 10, by + bh - 10, 15):
            for wx in range(bx + 6, bx + bw - 6, 10):
                if (wx + wy) % 5 == 0:
                    cv2.rectangle(img, (wx, wy), (wx + 4, wy + 8), 85, -1)

    # Add subtle Gaussian noise to simulate real camera sensor in low light
    noise = np.random.normal(0, 5, (h, w)).astype(np.int16)
    img_noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    path = os.path.join(DATA_DIR, "quiz2_low_contrast.jpg")
    cv2.imwrite(path, img_noisy)
    print(f"Generated: {path}")


def create_quiz3_document_images():
    """
    Generate tilted document images for perspective transformation and automated keystone correction.
    Produces normal tilt, varying angles (30, 45, 60, 75 deg), and cluttered background.
    """
    # 1. Create a pristine high-res A4 document
    doc_w, doc_h = 480, 640
    doc = np.full((doc_h, doc_w, 3), 245, dtype=np.uint8)

    # Document border / header
    cv2.rectangle(doc, (20, 20), (doc_w - 20, doc_h - 20), (30, 30, 30), 2)
    cv2.rectangle(doc, (30, 30), (doc_w - 30, 110), (180, 100, 30), -1)  # Header banner
    cv2.putText(doc, "NATIONAL YANG MING CHIAO TUNG UNIV.", (45, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(doc, "Multi-Modality Image Processing (MMIP)", (45, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # Document body text lines
    lines = [
        "Assignment 01: Computer Vision Analysis",
        "Student ID: 315833033",
        "Topic: Perspective Transformation & Keystone Correction",
        "",
        "Abstract:",
        "Perspective distortion occurs when planar objects are imaged from",
        "non-orthogonal viewpoints. A 3x3 homography matrix H with 8 degrees",
        "of freedom enables mapping from projective space to canonical Euclidean",
        "coordinates. By identifying four corresponding points, the original",
        "aspect ratio and text legibility can be fully restored.",
        "",
        "Key Equations:",
        "[u, v, 1]^T ~ H * [x, y, 1]^T",
        "h_33 = 1 (Standard normalization)",
        "",
        "Automated Pipeline:",
        "1. Bilateral Filtering to preserve edges while smoothing noise.",
        "2. Canny Edge Detection & Morphological Closing.",
        "3. Contour Analysis & Douglas-Peucker Polygon Approximation.",
        "4. Perspective Warp & Dynamic Aspect Ratio Rectification."
    ]

    y_offset = 145
    for line in lines:
        if line.startswith("Assignment") or line.startswith("Abstract") or line.startswith("Key") or line.startswith("Automated"):
            cv2.putText(doc, line, (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 120), 2)
        else:
            cv2.putText(doc, line, (40, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1)
        y_offset += 24

    # Save original ground truth document
    gt_path = os.path.join(DATA_DIR, "quiz3_doc_ground_truth.jpg")
    cv2.imwrite(gt_path, doc)

    # 2. Warp document into background scenes at various angles
    angles = [25, 30, 45, 60, 75]
    src_pts = np.float32([[0, 0], [doc_w, 0], [doc_w, doc_h], [0, doc_h]])

    for angle in angles:
        bg_w, bg_h = 1000, 800
        # Realistic wooden desk background
        bg = np.zeros((bg_h, bg_w, 3), dtype=np.uint8)
        for y in range(bg_h):
            wood_val = int(80 + 20 * np.sin(y / 15.0) + 15 * np.sin(y / 4.0))
            bg[y, :] = (wood_val // 2, wood_val * 2 // 3, wood_val)

        # Calculate perspective destination points based on tilt angle
        # Fore-shortening increases drastically as angle approaches 75 deg
        rad = np.radians(angle)
        scale_top = np.cos(rad) * 0.85
        center_x, center_y = bg_w // 2, bg_h // 2 + 30

        top_w = doc_w * scale_top * 0.75
        bot_w = doc_w * 0.95
        proj_h = doc_h * np.cos(rad * 0.7) * 0.85

        tl = [center_x - top_w // 2 + 40 * np.sin(rad), center_y - proj_h // 2]
        tr = [center_x + top_w // 2 + 40 * np.sin(rad), center_y - proj_h // 2 - 20 * np.sin(rad)]
        br = [center_x + bot_w // 2 - 20 * np.sin(rad), center_y + proj_h // 2]
        bl = [center_x - bot_w // 2 - 20 * np.sin(rad), center_y + proj_h // 2 + 10 * np.sin(rad)]

        dst_pts = np.float32([tl, tr, br, bl])

        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_doc = cv2.warpPerspective(doc, H, (bg_w, bg_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        # Mask for document
        mask = cv2.warpPerspective(np.ones((doc_h, doc_w), dtype=np.uint8) * 255, H, (bg_w, bg_h))
        mask_inv = cv2.bitwise_not(mask)

        # Composite document onto wooden desk
        bg_comp = cv2.bitwise_and(bg, bg, mask=mask_inv)
        final_img = cv2.add(bg_comp, warped_doc)

        # If normal tilt (25 deg), save as default quiz3_tilted_doc.jpg
        if angle == 25:
            fname = "quiz3_tilted_doc_normal.jpg"
            # Also save coordinates for manual baseline verification
            np.save(os.path.join(DATA_DIR, "quiz3_true_corners.npy"), dst_pts)
        else:
            fname = f"quiz3_tilted_doc_{angle}deg.jpg"

        cv2.imwrite(os.path.join(DATA_DIR, fname), final_img)
        print(f"Generated: {fname}")

    # 3. Create a cluttered background version (stationery, shadow) for advanced robustness test
    clutter = cv2.imread(os.path.join(DATA_DIR, "quiz3_tilted_doc_normal.jpg"))
    # Add a coffee cup
    cv2.circle(clutter, (180, 160), 45, (40, 40, 40), -1)
    cv2.circle(clutter, (180, 160), 38, (30, 70, 110), -1)  # coffee liquid
    # Add a pen
    cv2.line(clutter, (820, 200), (750, 680), (30, 30, 220), 8)
    cv2.imwrite(os.path.join(DATA_DIR, "quiz3_tilted_doc_cluttered.jpg"), clutter)
    print("Generated: quiz3_tilted_doc_cluttered.jpg")


def create_quiz4_stitching_pair():
    """
    Generate an overlapping panoramic image pair with rich texture and keypoints.
    Produces left and right camera views with ~45% horizontal overlap, plus stress test variants.
    """
    total_w, h = 1400, 600
    panorama = np.zeros((h, total_w, 3), dtype=np.uint8)

    # Beautiful architectural campus scene with distinct structures
    # Sky
    for y in range(250):
        val = int(220 - y * 0.4)
        panorama[y, :] = (val, val - 20, val - 60)

    # Distant skyline
    for x in range(0, total_w, 40):
        bh = 180 + int(70 * np.sin(x / 60.0) + 30 * np.cos(x / 15.0))
        cv2.rectangle(panorama, (x, 250 - bh), (x + 35, 250), (140, 130, 120), -1)

    # Ground
    for y in range(250, h):
        val = int(60 + (y - 250) * 0.3)
        panorama[y, :] = (val, val + 20, val)

    # Main Building 1 (Left to Center)
    cv2.rectangle(panorama, (150, 150), (550, 500), (90, 110, 170), -1)
    cv2.rectangle(panorama, (140, 140), (560, 160), (50, 60, 90), -1)  # Roof
    # Windows grid
    for wy in range(180, 480, 40):
        for wx in range(180, 530, 45):
            cv2.rectangle(panorama, (wx, wy), (wx + 25, wy + 25), (220, 220, 180), -1)
            cv2.rectangle(panorama, (wx, wy), (wx + 25, wy + 25), (30, 30, 30), 1)

    # Main Building 2 (Center to Right - Overlap Zone!)
    cv2.rectangle(panorama, (500, 120), (950, 520), (130, 100, 80), -1)
    cv2.rectangle(panorama, (480, 100), (970, 130), (70, 50, 40), -1)
    for wy in range(150, 490, 45):
        for wx in range(530, 920, 50):
            cv2.rectangle(panorama, (wx, wy), (wx + 30, wy + 30), (200, 230, 240), -1)
            cv2.rectangle(panorama, (wx, wy), (wx + 30, wy + 30), (40, 40, 40), 1)

    # Distinct Landmark Clocktower (Crucial landmark in overlap region: x=650 to 730)
    cv2.rectangle(panorama, (660, 60), (720, 300), (60, 70, 90), -1)
    cv2.circle(panorama, (690, 110), 22, (240, 240, 240), -1)  # Clock face
    cv2.circle(panorama, (690, 110), 22, (20, 20, 20), 2)
    cv2.line(panorama, (690, 110), (690, 95), (20, 20, 20), 2)  # Hour hand
    cv2.line(panorama, (690, 110), (702, 110), (20, 20, 20), 2)  # Minute hand

    # Rightside pavilion (x=950 to 1300)
    cv2.rectangle(panorama, (1000, 200), (1320, 480), (100, 140, 110), -1)
    for wy in range(230, 460, 50):
        for wx in range(1030, 1290, 60):
            cv2.rectangle(panorama, (wx, wy), (wx + 35, wy + 35), (240, 210, 180), -1)

    # Add foreground trees and lamp posts
    for tx in [80, 450, 750, 1150]:
        cv2.line(panorama, (tx, 450), (tx, 550), (30, 50, 70), 8)  # Trunk
        cv2.circle(panorama, (tx, 430), 45, (30, 120, 40), -1)  # Foliage
        cv2.circle(panorama, (tx - 15, 410), 30, (40, 150, 50), -1)

    # Split into Left and Right camera shots with ~45% overlap
    # Left view: x in [0, 800]
    # Right view: x in [500, 1300]
    # Overlap interval: x in [500, 800] (width = 300 / 800 = 37.5% to 45%)
    left_img = panorama[:, 0:800].copy()
    right_img = panorama[:, 520:1320].copy()

    # Add slight camera homography tilt to Right Image to simulate realistic hand-held perspective
    # (small rotation + slight scale variation)
    rh, rw = right_img.shape[:2]
    M_affine = cv2.getRotationMatrix2D((rw // 2, rh // 2), 2.5, 0.98)
    right_img = cv2.warpAffine(right_img, M_affine, (rw, rh), borderMode=cv2.BORDER_REFLECT)

    # 模擬真實相機雙視角拍攝常見之自動曝光/測光差異 (約 18% 曝光落差)
    # 使基礎題之覆蓋硬接縫 (Hard Seam) 清晰可辨，進階題羽化融合之消縫成效一目了然
    right_img = np.clip(right_img.astype(np.float32) * 0.82 + 10, 0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(DATA_DIR, "quiz4_scene_left.jpg"), left_img)
    cv2.imwrite(os.path.join(DATA_DIR, "quiz4_scene_right.jpg"), right_img)
    print("Generated: quiz4_scene_left.jpg & quiz4_scene_right.jpg")

    # Generate variants for Quiz 4 Advanced Stress Testing:
    # 1. Dark right image (-40% brightness)
    dark_right = np.clip(right_img.astype(np.float32) * 0.6, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(DATA_DIR, "quiz4_scene_right_dark40.jpg"), dark_right)

    # 2. Bright right image (+40% brightness)
    bright_right = np.clip(right_img.astype(np.float32) * 1.4, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(DATA_DIR, "quiz4_scene_right_bright40.jpg"), bright_right)

    # 3. Large rotation right image (25 degrees)
    M_rot = cv2.getRotationMatrix2D((rw // 2, rh // 2), 25.0, 1.0)
    rot25_right = cv2.warpAffine(right_img, M_rot, (rw, rh), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    cv2.imwrite(os.path.join(DATA_DIR, "quiz4_scene_right_rot25.jpg"), rot25_right)

    # 4. Extreme rotation right image (50 degrees - SIFT handles this due to rotation invariance)
    M_rot50 = cv2.getRotationMatrix2D((rw // 2, rh // 2), 50.0, 1.0)
    rot50_right = cv2.warpAffine(right_img, M_rot50, (rw, rh), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    cv2.imwrite(os.path.join(DATA_DIR, "quiz4_scene_right_rot50.jpg"), rot50_right)

    # 5. Low overlap pair (narrow overlap interval)
    low_overlap_right = panorama[:, 720:1400].copy()
    cv2.imwrite(os.path.join(DATA_DIR, "quiz4_scene_right_low_overlap.jpg"), low_overlap_right)
    print("Generated all Quiz 4 stress test variants.")


if __name__ == "__main__":
    print("=== Generating Test Datasets for MMIP HW01 ===")
    create_quiz1_color_image()
    create_quiz2_low_contrast_image()
    create_quiz3_document_images()
    create_quiz4_stitching_pair()
    print("=== All Test Datasets Successfully Created! ===")
