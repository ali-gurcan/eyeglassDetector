import cv2
import numpy as np
from pathlib import Path

def main():
    img_path = "images/8.png"
    txt_path = "auto_labeled_dataset/labels/8.txt"
    
    if not Path(img_path).exists() or not Path(txt_path).exists():
        print("❌ Dosyalar bulunamadı!")
        return

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    out = img.copy()
    overlay = img.copy()
    
    with open(txt_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
            
        cls_id = int(parts[0])
        # Koordinatları al ve piksele çevir
        coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
        coords[:, 0] *= w
        coords[:, 1] *= h
        pts = coords.astype(np.int32)
        
        # Mavi renkle boya
        color = (255, 0, 0)
        cv2.fillPoly(overlay, [pts], color)
        cv2.drawContours(out, [pts], -1, color, 2)
        
    # Şeffaflık ekle
    alpha = 0.5
    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    
    output_name = "test_result_auto_labeled_8.png"
    cv2.imwrite(output_name, out)
    print(f"🎉 İşlem başarılı! '{output_name}' dosyası oluşturuldu.")

if __name__ == "__main__":
    main()
