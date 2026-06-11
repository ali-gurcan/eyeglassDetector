import os
import cv2
import torch
import torch.nn.functional as F_torch

# 1. Mac için MPS gizleme yaması:
if hasattr(torch.backends, 'mps'):
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False

import numpy as np
from PIL import Image
from pathlib import Path

# 2. pin_memory() Mac'te MPS referansı verip çöküyor. CUDA yoksa no-op:
_orig_pin_memory = torch.Tensor.pin_memory
def _safe_pin_memory(self, device=None):
    if torch.cuda.is_available():
        return _orig_pin_memory(self, device=device)
    return self
torch.Tensor.pin_memory = _safe_pin_memory

# 3. SAM3'ün _addmm_activation fused kernel'i T4/CPU'da BFloat16 çakışması yaratıyor.
#    Standart PyTorch işlemleriyle değiştiriyoruz:
import sam3.perflib.fused as _fused
import sam3.model.vitdet as _vitdet

def _safe_addmm_act(act_type, linear, x):
    x = F_torch.linear(x, linear.weight, linear.bias)
    if act_type == torch.nn.GELU:
        x = F_torch.gelu(x)
    elif act_type == torch.nn.ReLU:
        x = F_torch.relu(x)
    elif act_type == torch.nn.SiLU:
        x = F_torch.silu(x)
    return x

_fused.addmm_act = _safe_addmm_act
_vitdet.addmm_act = _safe_addmm_act

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
import argparse

PROMPTS = [
    {
        "id": 0, # frame
        "text": "plastic or metal borders surrounding the lenses",
        "neg": "glass, eye, skin"
    },
    {
        "id": 1, # lens
        "text": "clear glass or plastic lens area inside the frame rims",
        "neg": "skin, frame, background"
    }
]

def main():
    parser = argparse.ArgumentParser()
    
    # Colab'de miyiz kontrolü
    in_colab = os.path.exists("/content")
    default_in = "/content/drive/MyDrive/Colab Notebooks/images" if in_colab else "images"
    default_out = "/content/drive/MyDrive/Colab Notebooks/sam3_labeled_dataset" if in_colab else "sam3_labeled_dataset"
    
    parser.add_argument("--input", type=str, default=default_in)
    parser.add_argument("--output", type=str, default=default_out)
    args = parser.parse_args()

    INPUT_FOLDER = args.input
    OUTPUT_FOLDER = args.output

    os.makedirs(os.path.join(OUTPUT_FOLDER, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, 'labels'), exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
        print(f"\n✅ GPU AKTİF: {torch.cuda.get_device_name(0)}\n")
    else:
        device = "cpu"
        print(f"Using device: {device}")
    
    try:
        print("Loading SAM 3 Model...")
        # Colab'de bpe_path hatasını önlemek için manuel yol veriyoruz
        bpe_path = None
        if os.path.exists("/content/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"):
            bpe_path = "/content/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
            
        model = build_sam3_image_model(device=device, bpe_path=bpe_path)
        model = model.to(torch.float32)
        processor = Sam3Processor(model)
        print("✅ SAM 3 Model Loaded!")
    except Exception as e:
        print(f"\n[ERROR] Model yuklenemedi: {e}")
        return

    img_paths = list(Path(INPUT_FOLDER).glob("*.png")) + list(Path(INPUT_FOLDER).glob("*.jpg"))
    print(f"Found {len(img_paths)} images to process.")

    for idx, img_path in enumerate(img_paths):
        filename = img_path.name
        
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        
        inference_state = processor.set_image(image)
        yolo_lines = []

        for p_info in PROMPTS:
            full_prompt = f"{p_info['text']}. Exclude: {p_info['neg']}"
            output = processor.set_text_prompt(state=inference_state, prompt=full_prompt)
            masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
            
            # Maske yapısını ve skorları debug etmek için ilk resimde yazdıralım
            if idx == 0:
                print(f"  -> Prompt: {p_info['id']} | Bulunan Mask Sayısı: {len(masks)}")
                if len(scores) > 0:
                    print(f"  -> Max Score: {max(scores):.4f}")
            
            for i in range(len(masks)):
                if scores[i] > 0.50: # Eşik değerini geçici olarak 0.50'ye çektim
                    # Maskeyi 0 ve 255 binary formatına çevirelim (Logit/Boolean hatasını önlemek için)
                    mask_tensor = masks[i]
                    if mask_tensor.dtype == torch.bool:
                        mask_np = mask_tensor.cpu().numpy().squeeze().astype(np.uint8) * 255
                    else:
                        mask_np = (mask_tensor > 0.0).cpu().numpy().squeeze().astype(np.uint8) * 255
                        
                    if mask_np.ndim != 2:
                        continue
                        
                    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if idx == 0 and scores[i] > 0.50:
                            print(f"  -> Mask {i} Contour Area: {area}")
                            
                        if area < 500: 
                            continue
                            
                        epsilon = 0.002 * cv2.arcLength(cnt, True)
                        approx_cnt = cv2.approxPolyDP(cnt, epsilon, True)
                        
                        pts = approx_cnt.reshape(-1, 2)
                        yolo_pts = []
                        for px, py in pts:
                            nx = max(0.0, min(1.0, float(px) / w))
                            ny = max(0.0, min(1.0, float(py) / h))
                            yolo_pts.append(f"{nx:.6f} {ny:.6f}")
                        
                        yolo_lines.append(f"{p_info['id']} " + " ".join(yolo_pts))

        if yolo_lines:
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, 'images', filename), cv2.imread(str(img_path)))
            with open(os.path.join(OUTPUT_FOLDER, 'labels', f"{img_path.stem}.txt"), 'w') as f:
                f.write("\n".join(yolo_lines))
            print(f"[{idx+1}/{len(img_paths)}] ✅ {filename} - {len(yolo_lines)} etiket")
        else:
            print(f"[{idx+1}/{len(img_paths)}] ⏭️ {filename} - bos")

if __name__ == "__main__":
    main()
