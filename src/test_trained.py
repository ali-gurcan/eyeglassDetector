#!/usr/bin/env python3
"""
Test the trained YOLOv8 segmentation model on project images.
"""

import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO

INPUT_FOLDER = "images"
OUTPUT_FOLDER = "output_trained"
MODEL_PATH = "runs/segment/runs/segment/eyeglass/weights/best.pt"

COLORS = {
    "frame": (128, 0, 0),     # navy blue
    "glass": (0, 255, 255),   # cyan
}
CONTOUR_THICKNESS = 2

os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def main():
    model = YOLO(MODEL_PATH)
    print(f"Model: {MODEL_PATH}")

    image_exts = ("*.png", "*.jpg", "*.jpeg")
    image_paths = []
    for ext in image_exts:
        image_paths.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    image_paths.sort()

    print(f"Toplam {len(image_paths)} resim test edilecek...\n")

    total_glass = 0
    total_frame = 0
    success = 0

    for img_path in image_paths:
        fname = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            print(f"[HATA] {fname} -> Okunamadı")
            continue

        results = model.predict(img, conf=0.25, verbose=False)
        result = results[0]

        out = img.copy()
        overlay = img.copy()
        gc, fc = 0, 0

        if result.masks is not None:
            for i, mask in enumerate(result.masks.xy):
                cls_id = int(result.boxes.cls[i])
                cls_name = result.names[cls_id]
                conf = float(result.boxes.conf[i])

                color = COLORS.get(cls_name, (255, 255, 255))
                pts = np.array(mask, dtype=np.int32)

                if len(pts) < 3:
                    continue

                cv2.fillPoly(overlay, [pts], color)
                cv2.drawContours(out, [pts], -1, color, CONTOUR_THICKNESS)

                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                label = f"{cls_name} {conf:.0%}"
                cv2.putText(out, label, (cx - 30, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                if cls_name == "glass":
                    gc += 1
                elif cls_name == "frame":
                    fc += 1

        alpha = 0.3
        out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

        total_glass += gc
        total_frame += fc

        if gc > 0 or fc > 0:
            success += 1
            print(f"[OK]   {fname} -> glass={gc}, frame={fc}")
        else:
            print(f"[MISS] {fname} -> Hiç tespit yok")

        stem, ext = os.path.splitext(fname)
        out_path = os.path.join(OUTPUT_FOLDER, f"{stem}_trained{ext}")
        cv2.imwrite(out_path, out)

    print(f"\n--- Sonuçlar ---")
    print(f"  Başarılı: {success}/{len(image_paths)}")
    print(f"  Toplam glass: {total_glass}")
    print(f"  Toplam frame: {total_frame}")
    print(f"  Çıktılar: {OUTPUT_FOLDER}/")


if __name__ == "__main__":
    main()
