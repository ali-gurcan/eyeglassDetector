import os
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# --- Configuration ---
INPUT_FOLDER = 'images'
OUTPUT_FOLDER = 'sam3_labeled_dataset'
os.makedirs(os.path.join(OUTPUT_FOLDER, 'images'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_FOLDER, 'labels'), exist_ok=True)

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
    device = "mps" if torch.backends.mps.is_available() else "cpu"
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
            # Note: SAM 3 API might vary, assuming set_text_prompt as per README content
            # response = processor.set_text_prompt(state=inference_state, prompt=p_info["text"])
            # Some versions might support negative prompts in a different way or via a combined string
            full_prompt = p_info["text"]
            
            output = processor.set_text_prompt(state=inference_state, prompt=full_prompt)
            masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
            
            # Filter by score (e.g., > 0.5)
            for i in range(len(masks)):
                if scores[i] > 0.4:
                    mask = masks[i].cpu().numpy().astype(np.uint8)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for cnt in contours:
                        if cv2.contourArea(cnt) < 100: continue
                        
                        pts = cnt.reshape(-1, 2)
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
