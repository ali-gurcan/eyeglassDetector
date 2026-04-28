#!/usr/bin/env python3
"""
Eyeglass Lens Detection — SAM 2.1 + Edge Refinement Pipeline
-------------------------------------------------------------
Three-stage approach:
  1. glasses-detector → approximate bounding boxes for each lens
  2. SAM 2.1 Large → pixel-precise FRAME segmentation within each bbox
  3. Canny + contour hierarchy → actual LENS edge inside the frame

SAM 2 finds the frame region; then main.py's edge detection pipeline
(Canny + RETR_TREE hierarchy + filters) finds the real lens edge where
glass meets the frame groove.
"""

import glob
import os
import sys
import shutil
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

SAM2_REPO = "/tmp/sam2"
sys.path.insert(0, SAM2_REPO)

# --- Configuration ---
INPUT_FOLDER = 'images'
OUTPUT_FOLDER = 'output'
DEBUG_FOLDER = 'debug'
MODELS_DIR = Path(__file__).parent.parent / "models"
CONTOUR_COLOR = (255, 255, 0)
CONTOUR_THICKNESS = 2

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DEBUG_FOLDER, exist_ok=True)

DEBUG_MODE = False


def clear_output_dir():
    output_path = Path(OUTPUT_FOLDER)
    output_path.mkdir(parents=True, exist_ok=True)
    for item in output_path.iterdir():
        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except OSError:
            pass


def load_models():
    """Load glasses-detector (for bbox) and SAM 2.1 (for segmentation)."""
    from glasses_detector import GlassesSegmenter
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    gd = GlassesSegmenter(kind="lenses", size="medium")
    print("[INFO] glasses-detector yüklendi (bbox için)")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    ckpt = str(MODELS_DIR / "sam2.1_hiera_large.pt")

    sam2_model = build_sam2(cfg, ckpt, device=device)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    print(f"[INFO] SAM 2.1 Large yüklendi ({device})")

    return gd, sam2_predictor


def get_lens_bboxes(img_rgb: np.ndarray, gd_segmenter) -> list[list[int]]:
    """Use glasses-detector probability map to extract per-lens bounding boxes."""
    proba = gd_segmenter(img_rgb, format="proba").cpu().numpy()
    if proba.ndim > 2:
        proba = proba.squeeze()

    mask = (proba > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = img_rgb.shape[:2]
    total_area = h * w
    bboxes = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < total_area * 0.005:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        pad_x = int(bw * 0.05)
        pad_y = int(bh * 0.05)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)
        bboxes.append([x1, y1, x2, y2])

    return bboxes


def predict_sam2(predictor, img_rgb: np.ndarray,
                 bboxes: list[list[int]]) -> list[np.ndarray]:
    """Run SAM 2.1 segmentation for each lens bbox. Returns list of binary masks."""
    predictor.set_image(img_rgb)
    masks = []
    for box in bboxes:
        box_np = np.array(box, dtype=np.float32)
        pred_masks, scores, _ = predictor.predict(
            box=box_np,
            multimask_output=True,
        )
        best_idx = int(np.argmax(scores))
        masks.append(pred_masks[best_idx])
    return masks


def get_frame_contour(mask: np.ndarray):
    """Extract the main contour from a SAM 2 binary mask."""
    mask_u8 = mask.astype(np.uint8) * 255
    mask_u8 = cv2.GaussianBlur(mask_u8, (5, 5), 1)
    _, mask_u8 = cv2.threshold(mask_u8, 127, 255, cv2.THRESH_BINARY)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k)

    contours_raw, _ = cv2.findContours(
        mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    if not contours_raw:
        return None, mask_u8

    cnt = max(contours_raw, key=cv2.contourArea)
    total_area = mask.shape[0] * mask.shape[1]
    if cv2.contourArea(cnt) < total_area * 0.002:
        return None, mask_u8

    return cnt, mask_u8


# ---------------------------------------------------------------------------
# Stage 3: Lens Edge Detection (Canny + Contour Hierarchy)
# Directly from main.py's proven approach:
#   - Bilateral filter + CLAHE preprocessing
#   - Pupil region masking (suppress iris/pupil false edges)
#   - Canny edge detection with auto-threshold
#   - Morphological closing to bridge gaps
#   - RETR_TREE contour hierarchy (inner contour = lens edge)
#   - Area / aspect / solidity / position filters
#   - Shape completion with fitEllipse → convexHull
# ---------------------------------------------------------------------------

def refine_to_lens_edge(img_bgr: np.ndarray, frame_contour: np.ndarray,
                        frame_mask_u8: np.ndarray,
                        debug_prefix: str = '') -> np.ndarray:
    """
    Find the actual lens edge inside the SAM 2 frame mask using
    Canny edge detection + contour hierarchy analysis (from main.py).

    SAM 2 gives us the perfect frame region; within that region we run
    the same edge detection pipeline that main.py uses to find the real
    lens boundary where glass meets the frame groove.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # --- ROI from frame bounding box (generous padding for edge context) ---
    bx, by, bw, bh = cv2.boundingRect(frame_contour)
    pad = int(max(bw, bh) * 0.25)
    x1, y1 = max(0, bx - pad), max(0, by - pad)
    x2, y2 = min(w, bx + bw + pad), min(h, by + bh + pad)

    roi_gray = gray[y1:y2, x1:x2].copy()
    roi_mask = frame_mask_u8[y1:y2, x1:x2]
    rh, rw = roi_gray.shape

    # Frame center in ROI coordinates
    M = cv2.moments(frame_contour)
    if M['m00'] == 0:
        return frame_contour
    fcx = int(M['m10'] / M['m00']) - x1
    fcy = int(M['m01'] / M['m00']) - y1

    frame_area = cv2.contourArea(frame_contour)

    # Detection passes: (morph_iters, pupil_mask, canny_sigma, clahe_clip, mode)
    passes = [
        (1, False, 0.33, 2.0, 'normal'),
        (3, True,  0.33, 2.0, 'retry'),
        (2, False, 0.50, 4.0, 'relaxed'),
    ]

    for morph_iters, use_pupil_mask, sigma, clip, mode in passes:

        # --- Preprocessing (from main.py) ---
        blurred = cv2.bilateralFilter(roi_gray, 9, 75, 75)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)

        # --- Pupil region masking (from main.py) ---
        if use_pupil_mask:
            axes = (int(rw * 0.18), int(rh * 0.15))
            pmask = np.zeros_like(enhanced, dtype=np.float32)
            cv2.ellipse(pmask, (fcx, fcy), axes, 0, 0, 360, 1.0, -1)
            pmask = cv2.GaussianBlur(pmask, (21, 21), 11)
            heavy = cv2.GaussianBlur(enhanced, (99, 99), 30)
            enhanced = (enhanced * (1.0 - pmask) + heavy * pmask).astype(np.uint8)

        # --- Canny edge detection (from main.py: auto-threshold) ---
        v = np.median(enhanced)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(enhanced, lower, upper)

        # --- Morphological closing (from main.py) ---
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel,
                                  iterations=morph_iters)

        # --- Find contours with hierarchy (from main.py: RETR_TREE) ---
        # No mask clipping here — the full hierarchy must form naturally
        # so that frame outer edge = parent, lens edge = inner child.
        contours, hierarchy = cv2.findContours(
            closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours or hierarchy is None:
            continue

        hier = hierarchy[0]

        # --- Filter thresholds (from main.py: normal vs relaxed) ---
        if mode == 'normal':
            min_area_r, max_area_r = 0.15, 0.92
            min_asp, max_asp = 0.5, 2.5
            min_sol = 0.80
        else:
            min_area_r, max_area_r = 0.08, 0.95
            min_asp, max_asp = 0.3, 3.5
            min_sol = 0.65

        candidates = []

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < frame_area * min_area_r or area > frame_area * max_area_r:
                continue

            cx_c, cy_c, cw, ch = cv2.boundingRect(cnt)
            aspect = float(cw) / ch if ch > 0 else 0
            if not (min_asp < aspect < max_asp):
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-6)
            if solidity < min_sol:
                continue

            # Center must fall inside the frame mask (replaces bitwise_and)
            ccx, ccy = cx_c + cw // 2, cy_c + ch // 2
            if 0 <= ccy < rh and 0 <= ccx < rw:
                if roi_mask[ccy, ccx] == 0:
                    continue
            else:
                continue

            dist = np.sqrt((ccx - fcx) ** 2 + (ccy - fcy) ** 2)
            if dist > max(rw, rh) * 0.4:
                continue

            # Inner contour priority (from main.py)
            is_inner = hier[i][3] != -1

            # Bottom-anchor scoring (from main.py)
            y_max = cnt[:, :, 1].max()
            bottomness = (y_max / rh) ** 2

            score = area * bottomness
            if is_inner:
                score *= 3.0

            candidates.append((score, cnt))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_cnt = candidates[0][1]

        # --- Shape completion (from main.py) ---
        if len(best_cnt) >= 5:
            try:
                ell = cv2.fitEllipse(best_cnt)
                center_e = (int(ell[0][0]), int(ell[0][1]))
                axes_e = (int(ell[1][0] / 2), int(ell[1][1] / 2))
                angle_e = int(ell[2])
                hull = cv2.ellipse2Poly(center_e, axes_e, angle_e, 0, 360, 5)
                hull = hull.reshape(-1, 1, 2)
            except Exception:
                hull = cv2.convexHull(best_cnt)
        else:
            hull = cv2.convexHull(best_cnt)

        # Offset back to full image coordinates
        hull = hull + [x1, y1]

        if DEBUG_MODE and debug_prefix:
            n_cand = len(candidates)
            print(f"    [LENS] {debug_prefix}: {mode} ({n_cand} candidates, "
                  f"area={cv2.contourArea(best_cnt):.0f})")

        return hull

    # All passes failed — fall back to frame contour
    if DEBUG_MODE and debug_prefix:
        print(f"    [LENS] {debug_prefix}: FAILED — using frame contour")
    return frame_contour


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_image(img_path: str, gd_segmenter, sam2_predictor) -> tuple:
    """
    Full pipeline for one image:
      1. Read & resize
      2. glasses-detector → bounding boxes
      3. SAM 2.1 → frame masks (çerçeve kenarı)
      4. Canny + hierarchy → lens edges (cam kenarı)
    Returns (img_bgr, num_contours, lens_contours, frame_contours, combined_mask).
    """
    input_path = Path(img_path)
    img = cv2.imread(str(input_path))

    if img is None:
        return None, 0, [], [], None

    if img.shape[1] > 1000:
        scale = 1000 / img.shape[1]
        img = cv2.resize(img, None, fx=scale, fy=scale)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    bboxes = get_lens_bboxes(img_rgb, gd_segmenter)
    if not bboxes:
        return img, 0, [], [], np.zeros((h, w), dtype=np.uint8)

    sam2_masks = predict_sam2(sam2_predictor, img_rgb, bboxes)

    lens_contours = []
    frame_contours = []
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    for i, mask in enumerate(sam2_masks):
        combined_mask |= mask.astype(np.uint8) * 255
        frame_cnt, mask_u8 = get_frame_contour(mask)

        if frame_cnt is None:
            continue

        frame_contours.append(frame_cnt)

        debug_prefix = f"{input_path.stem}_lens{i}" if DEBUG_MODE else ''
        lens_cnt = refine_to_lens_edge(img, frame_cnt, mask_u8, debug_prefix)
        lens_contours.append(lens_cnt)

    if DEBUG_MODE:
        print(f"    [DEBUG] BBox: {len(bboxes)}, SAM2 maske: {len(sam2_masks)}, "
              f"Çerçeve: {len(frame_contours)}, Cam: {len(lens_contours)}")

    return img, len(lens_contours), lens_contours, frame_contours, combined_mask


def draw_and_save(img, lens_contours, frame_contours, input_path: Path):
    """Draw lens edge (yellow) and optionally frame edge (green) on image."""
    # Lens edge — primary output (yellow, thick)
    for cnt in lens_contours:
        pts = cnt.reshape(-1, 2)
        for i in range(len(pts)):
            p1 = tuple(pts[i])
            p2 = tuple(pts[(i + 1) % len(pts)])
            cv2.line(img, p1, p2, CONTOUR_COLOR, CONTOUR_THICKNESS, cv2.LINE_AA)

    # Frame edge — debug reference (green, thin)
    if DEBUG_MODE:
        for cnt in frame_contours:
            pts = cnt.reshape(-1, 2)
            for i in range(len(pts)):
                p1 = tuple(pts[i])
                p2 = tuple(pts[(i + 1) % len(pts)])
                cv2.line(img, p1, p2, (0, 200, 0), 1, cv2.LINE_AA)

    output_path = Path(OUTPUT_FOLDER) / f"{input_path.stem}_model{input_path.suffix}"
    cv2.imwrite(str(output_path), img)
    return output_path


def run_fallback(img_path: str):
    """Fall back to the traditional CV pipeline from main.py."""
    try:
        from main import process_image as cv_process, load_face_cascade, load_eye_cascade
        face_cascade = load_face_cascade()
        eye_cascade = load_eye_cascade()
        cv_process(img_path, face_cascade, eye_cascade)
        return True
    except Exception as e:
        print(f"    [WARN] Fallback hatası: {e}")
        return False


def save_debug_images(img, mask, bboxes, lens_contours, frame_contours, input_path: Path):
    """Save comprehensive debug visualizations."""
    # Frame mask overlay
    mask_colored = np.zeros_like(img)
    mask_colored[mask > 0] = (0, 255, 255)
    blended = cv2.addWeighted(img, 0.6, mask_colored, 0.4, 0)

    for box in bboxes:
        cv2.rectangle(blended, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    debug_path = Path(DEBUG_FOLDER)
    cv2.imwrite(str(debug_path / f"{input_path.stem}_frame_mask.png"), blended)

    # Side-by-side: frame edge (green) vs lens edge (yellow)
    compare = img.copy()
    for cnt in frame_contours:
        cv2.drawContours(compare, [cnt], -1, (0, 255, 0), 2)
    for cnt in lens_contours:
        cv2.drawContours(compare, [cnt], -1, (0, 255, 255), 2)
    cv2.imwrite(str(debug_path / f"{input_path.stem}_frame_vs_lens.png"), compare)

    # Raw mask
    cv2.imwrite(str(debug_path / f"{input_path.stem}_mask_raw.png"), mask)


def main():
    global DEBUG_MODE

    parser = argparse.ArgumentParser(
        description='Eyeglass Lens Edge Detection — SAM 2.1 + Edge Refinement'
    )
    parser.add_argument(
        '--debug', action='store_true',
        help='Debug modu: ara görüntüleri ve detaylı logları kaydet'
    )
    parser.add_argument(
        '--fallback', action='store_true',
        help='Model sonuç üretmezse geleneksel CV pipeline\'ına düş'
    )
    args = parser.parse_args()

    DEBUG_MODE = args.debug

    if DEBUG_MODE:
        print("--- SAM 2.1 + Lens Edge Pipeline (DEBUG) ---")
    else:
        print("--- SAM 2.1 + Lens Edge Pipeline ---")

    gd_segmenter, sam2_predictor = load_models()
    clear_output_dir()

    if DEBUG_MODE:
        debug_path = Path(DEBUG_FOLDER)
        if debug_path.exists():
            for item in debug_path.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                except OSError:
                    pass

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
        image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

    if not image_files:
        print(f"[WARN] '{INPUT_FOLDER}/' klasöründe resim bulunamadı.")
        return

    print(f"Toplam {len(image_files)} resim işlenecek...\n")

    success_count = 0
    fallback_count = 0
    fail_count = 0

    for img_path in image_files:
        input_path = Path(img_path)
        filename = input_path.name

        result = process_image(img_path, gd_segmenter, sam2_predictor)
        img, num_contours, lens_contours, frame_contours, mask = result

        if img is None:
            print(f"[HATA] {filename} -> Görüntü okunamadı.")
            fail_count += 1
            continue

        if DEBUG_MODE and mask is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            bboxes = get_lens_bboxes(img_rgb, gd_segmenter)
            save_debug_images(img.copy(), mask, bboxes,
                              lens_contours, frame_contours, input_path)

        if num_contours > 0:
            draw_and_save(img, lens_contours, frame_contours, input_path)
            print(f"[OK] {filename} -> {num_contours} cam kenarı tespit edildi")
            success_count += 1
        elif args.fallback:
            print(f"[FALLBACK] {filename} -> Tespit yok, CV pipeline deneniyor...")
            if run_fallback(img_path):
                fallback_count += 1
            else:
                fail_count += 1
        else:
            output_path = Path(OUTPUT_FOLDER) / f"{input_path.stem}_model{input_path.suffix}"
            cv2.imwrite(str(output_path), img)
            print(f"[UYARI] {filename} -> Cam kenarı tespit edilemedi.")
            fail_count += 1

    print(f"\n--- Bitti ---")
    print(f"  Cam kenarı tespit: {success_count}")
    if args.fallback:
        print(f"  Fallback (CV):     {fallback_count}")
    print(f"  Başarısız:         {fail_count}")
    print(f"  Toplam:            {len(image_files)}")

    if DEBUG_MODE:
        print(f"\n[DEBUG] Debug görüntüleri '{DEBUG_FOLDER}/' klasöründe.")


if __name__ == "__main__":
    main()
