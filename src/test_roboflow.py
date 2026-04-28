#!/usr/bin/env python3
"""
Test Roboflow eyeglass segmentation model (azaduni/eyeglass-6wu5y).
Sends images to the Roboflow hosted inference API and draws
glass/frame segmentation masks.

Usage:
  export ROBOFLOW_API_KEY="your_key_here"
  python src/test_roboflow.py
"""

import base64
import glob
import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

import cv2
import numpy as np

INPUT_FOLDER = "images"
OUTPUT_FOLDER = "output_roboflow"
ROBOFLOW_MODEL = "eyeglass-6wu5y"
ROBOFLOW_VERSION = "1"

COLORS = {
    "glass": (0, 255, 255),   # cyan
    "frame": (128, 0, 0),     # navy blue (lacivert)
}
CONTOUR_THICKNESS = 2

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def load_env():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip())


def get_api_key():
    load_env()
    key = os.environ.get("ROBOFLOW_API_KEY", "")
    if not key or key == "your_api_key_here":
        print("[HATA] ROBOFLOW_API_KEY ayarlanmamış.")
        print("  1. .env dosyasını aç")
        print("  2. ROBOFLOW_API_KEY=senin_keyin şeklinde güncelle")
        print("  (Ücretsiz key: https://app.roboflow.com > Settings > API Key)")
        sys.exit(1)
    return key


def infer_image(image_path: str, api_key: str, orig_hw: tuple) -> dict:
    """Send image to Roboflow API. Returns (response_dict, scale_x, scale_y)."""
    img = cv2.imread(image_path)
    if img is None:
        return {}, 1.0, 1.0

    orig_h, orig_w = orig_hw
    max_dim = 1024
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))

    sent_h, sent_w = img.shape[:2]
    scale_x = orig_w / sent_w
    scale_y = orig_h / sent_h

    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    img_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

    url = (
        f"https://outline.roboflow.com/{ROBOFLOW_MODEL}/{ROBOFLOW_VERSION}"
        f"?api_key={api_key}"
    )

    req = urllib.request.Request(
        url,
        data=img_b64.encode("utf-8"),
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8")), scale_x, scale_y
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"  [API HATA] {e.code}: {body[:200]}")
        return {}, scale_x, scale_y
    except Exception as e:
        print(f"  [API HATA] {e}")
        return {}, scale_x, scale_y


def draw_predictions(img_bgr: np.ndarray, predictions: list,
                     scale_x: float = 1.0, scale_y: float = 1.0):
    result = img_bgr.copy()
    overlay = img_bgr.copy()

    glass_count = 0
    frame_count = 0

    for pred in predictions:
        cls = pred.get("class", "unknown")
        confidence = pred.get("confidence", 0)
        points = pred.get("points", [])

        if not points or confidence < 0.25:
            continue

        color = COLORS.get(cls, (255, 255, 255))
        pts = np.array(
            [[int(p["x"] * scale_x), int(p["y"] * scale_y)] for p in points],
            dtype=np.int32,
        )

        cv2.fillPoly(overlay, [pts], color)
        cv2.drawContours(result, [pts], -1, color, CONTOUR_THICKNESS)

        if cls == "glass":
            glass_count += 1
        elif cls == "frame":
            frame_count += 1

        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        label = f"{cls} {confidence:.0%}"
        cv2.putText(result, label, (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    alpha = 0.3
    result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

    return result, glass_count, frame_count


def main():
    api_key = get_api_key()

    image_exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    image_paths = []
    for ext in image_exts:
        image_paths.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    image_paths.sort()

    if not image_paths:
        print(f"[HATA] '{INPUT_FOLDER}' klasöründe görüntü bulunamadı.")
        sys.exit(1)

    print(f"--- Roboflow Eyeglass Segmentation Test ---")
    print(f"Model: {ROBOFLOW_MODEL}/v{ROBOFLOW_VERSION}")
    print(f"Toplam {len(image_paths)} resim test edilecek...\n")

    total_glass = 0
    total_frame = 0
    success = 0
    failed = 0

    for img_path in image_paths:
        fname = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[HATA] {fname} -> Okunamadı")
            failed += 1
            continue

        orig_hw = img.shape[:2]
        response, sx, sy = infer_image(img_path, api_key, orig_hw)
        predictions = response.get("predictions", [])

        if not predictions:
            print(f"[MISS] {fname} -> Hiç tespit yok")
            failed += 1
            out = img.copy()
            cv2.putText(out, "No detection", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            out, gc, fc = draw_predictions(img, predictions, sx, sy)
            total_glass += gc
            total_frame += fc
            print(f"[OK]   {fname} -> glass={gc}, frame={fc}, "
                  f"total_preds={len(predictions)}")
            success += 1

        stem, ext = os.path.splitext(fname)
        out_path = os.path.join(OUTPUT_FOLDER, f"{stem}_roboflow{ext}")
        cv2.imwrite(out_path, out)

    print(f"\n--- Sonuçlar ---")
    print(f"  Başarılı: {success}/{len(image_paths)}")
    print(f"  Başarısız: {failed}/{len(image_paths)}")
    print(f"  Toplam glass tespiti: {total_glass}")
    print(f"  Toplam frame tespiti: {total_frame}")
    print(f"  Çıktılar: {OUTPUT_FOLDER}/")


if __name__ == "__main__":
    main()
