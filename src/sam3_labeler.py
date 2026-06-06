import os
import cv2
import torch
# SAM3 Processor'ın kendi içinde MPS'i kontrol edip bazı tensörleri GPU'ya atmasını 
# (ve dolayısıyla weight-input uyuşmazlığını) engellemek için MPS'i PyTorch seviyesinde gizliyoruz:
if hasattr(torch.backends, 'mps'):
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False

import numpy as np
from PIL import Image
from pathlib import Path
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

import argparse

# --- Configuration ---
# Klasör yolları artık main() içerisinde terminal argümanı olarak (veya varsayılan olarak) belirlenecek.

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
    parser.add_argument("--input", type=str, default="images", help="Girdi görsellerinin klasör yolu")
    parser.add_argument("--output", type=str, default="sam3_labeled_dataset", help="Çıktı klasör yolu")
    args = parser.parse_args()

    INPUT_FOLDER = args.input
    OUTPUT_FOLDER = args.output

    os.makedirs(os.path.join(OUTPUT_FOLDER, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, 'labels'), exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda" # Colab T4 GPU için
    else:
        # Apple Silicon (MPS) üzerindeki PyTorch uyumsuzluğunu önlemek için Mac'te mecburen CPU
        device = "cpu"
        
    print(f"Using device: {device}")
    
    try:
        print("Loading SAM 3 Model (This may download weights if authenticated)...")
        model = build_sam3_image_model(device=device)
        processor = Sam3Processor(model)
        print("SAM 3 Model Loaded!")
    except Exception as e:
        print(f"\n[ERROR] Model yuklenemedi: {e}")
        print("Lutfen Hugging Face uzerinden 'facebook/sam3' erisimi aldiginizdan ve 'hf auth login' yaptiginizdan emin olun.")
        return

    img_paths = list(Path(INPUT_FOLDER).glob("*.png")) + list(Path(INPUT_FOLDER).glob("*.jpg"))
    print(f"Found {len(img_paths)} images to process.")

    for img_path in img_paths:
        filename = img_path.name
        print(f"Processing {filename}...")
        
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        
        inference_state = processor.set_image(image)
        yolo_lines = []

        for p_info in PROMPTS:
            # Negatif kelimeleri de metne dahil ederek modeli daha net yönlendiriyoruz
            full_prompt = f"{p_info['text']}. Exclude: {p_info['neg']}"
            
            output = processor.set_text_prompt(state=inference_state, prompt=full_prompt)
            masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
            
            # Etiket kalitesini artırmak için düşük güven skorlu (score < 0.65) sonuçları elliyoruz
            for i in range(len(masks)):
                if scores[i] > 0.65:
                    mask = masks[i].cpu().numpy().astype(np.uint8)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for cnt in contours:
                        # Ufak gürültüleri (noise) etiketlememek için minimum alanı artırdık (örneğin 500)
                        if cv2.contourArea(cnt) < 500: continue
                        
                        # Poligonu daha pürüzsüz ve temiz hale getirmek için yaklaşıklaştırma (smoothing)
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
            # Save
            cv2.imwrite(os.path.join(OUTPUT_FOLDER, 'images', filename), cv2.imread(str(img_path)))
            with open(os.path.join(OUTPUT_FOLDER, 'labels', f"{img_path.stem}.txt"), 'w') as f:
                f.write("\n".join(yolo_lines))
            print(f"  [OK] {filename} labeled with {len(yolo_lines)} components.")
        else:
            print(f"  [SKIP] No components found for {filename}")

if __name__ == "__main__":
    main()
