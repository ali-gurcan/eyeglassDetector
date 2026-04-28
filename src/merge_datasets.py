import os
import shutil
from pathlib import Path

# --- Configuration ---
ROBOFLOW_DATASET = 'EyeGlassDetection.v1i.yolov8.dataset'
AUTO_DATASET = 'auto_labeled_dataset'
MERGED_DATASET = 'glass_merged_dataset'

def merge():
    print(f"Creating merged dataset at {MERGED_DATASET}...")
    
    # 1. Clean and create directories
    if os.path.exists(MERGED_DATASET):
        shutil.rmtree(MERGED_DATASET)
    
    os.makedirs(MERGED_DATASET, exist_ok=True)
    
    # 2. Copy Roboflow dataset structure
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(ROBOFLOW_DATASET, split)
        if os.path.exists(split_path):
            shutil.copytree(split_path, os.path.join(MERGED_DATASET, split))
            print(f"  Copied {split} split from Roboflow dataset.")

    # 3. Copy data.yaml
    shutil.copy(os.path.join(ROBOFLOW_DATASET, 'data.yaml'), os.path.join(MERGED_DATASET, 'data.yaml'))
    
    # 4. Add auto-labeled dataset to TRAIN split
    auto_images = os.path.join(AUTO_DATASET, 'images')
    auto_labels = os.path.join(AUTO_DATASET, 'labels')
    
    target_images = os.path.join(MERGED_DATASET, 'train', 'images')
    target_labels = os.path.join(MERGED_DATASET, 'train', 'labels')
    
    os.makedirs(target_images, exist_ok=True)
    os.makedirs(target_labels, exist_ok=True)
    
    count = 0
    if os.path.exists(auto_images):
        for img_file in os.listdir(auto_images):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Copy image
                shutil.copy(os.path.join(auto_images, img_file), os.path.join(target_images, f"auto_{img_file}"))
                
                # Copy label if exists
                label_file = os.path.splitext(img_file)[0] + ".txt"
                if os.path.exists(os.path.join(auto_labels, label_file)):
                    shutil.copy(os.path.join(auto_labels, label_file), os.path.join(target_labels, f"auto_{label_file}"))
                count += 1
    
    print(f"  Added {count} images/labels from auto-labeled dataset to train split.")

    # 5. Update data.yaml paths (Roboflow exports often use relative paths like ../train/images)
    # We want to make them local to the merged folder if possible, but YOLO works with both.
    # Let's check and fix if needed.
    with open(os.path.join(MERGED_DATASET, 'data.yaml'), 'r') as f:
        lines = f.readlines()
    
    with open(os.path.join(MERGED_DATASET, 'data.yaml'), 'w') as f:
        for line in lines:
            if line.startswith('train:'):
                f.write('train: train/images\n')
            elif line.startswith('val:'):
                f.write('val: valid/images\n')
            elif line.startswith('test:'):
                f.write('test: test/images\n')
            else:
                f.write(line)
    
    print("Done! Combined dataset ready at 'glass_merged_dataset'.")

if __name__ == "__main__":
    merge()
