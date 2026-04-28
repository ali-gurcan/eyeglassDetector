#!/usr/bin/env python3
"""
Train YOLOv8 segmentation model on eyeglass dataset.

Usage:
  python src/train.py
"""

from pathlib import Path
from ultralytics import YOLO

DATASET_DIR = Path(__file__).parent.parent / "dataset" / "eyeglass.v1i.yolov8"
DATA_YAML = DATASET_DIR / "data.yaml"

MODEL_BASE = "yolov8n-seg.pt"
EPOCHS = 150
IMG_SIZE = 640
BATCH = 8
DEVICE = "mps"

PROJECT = "runs"
NAME = "eyeglass"


def main():
    print(f"Dataset: {DATASET_DIR}")
    print(f"Base model: {MODEL_BASE}")
    print(f"Epochs: {EPOCHS}, ImgSize: {IMG_SIZE}, Batch: {BATCH}")
    print(f"Device: {DEVICE}\n")

    model = YOLO(MODEL_BASE)

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        device=DEVICE,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        # Augmentation (critical for small datasets)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        patience=30,
        save=True,
        save_period=25,
        verbose=True,
    )

    print("\n--- Eğitim tamamlandı ---")
    print(f"Best model: {PROJECT}/{NAME}/weights/best.pt")

    best = YOLO(f"{PROJECT}/{NAME}/weights/best.pt")
    metrics = best.val(data=str(DATA_YAML), device=DEVICE)
    print(f"\nValidation mAP50: {metrics.seg.map50:.4f}")
    print(f"Validation mAP50-95: {metrics.seg.map:.4f}")


if __name__ == "__main__":
    main()
