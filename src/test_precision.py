import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output_precision')
YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'runs/segment/runs/segment/eyeglass/weights/best.pt')

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("YOLO Modeli Yükleniyor...")
yolo_model = YOLO(YOLO_MODEL_PATH)
print("Model Yüklendi. İşlem başlıyor...")

def refine_with_image_processing(img_bgr, yolo_mask_u8, bbox):
    """
    imageProcessing.py'deki orijinal Canny + RETR_TREE mantığını kullanarak
    YOLO'nun bulduğu alanın içinde gerçek görüntünün kenarlarını arar.
    Matematiksel uydurma YAPMAZ, fotoğraftaki asıl piksel kırılmalarını bulur.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # BBox'ı biraz genişletelim ki Canny kenarları tam görebilsin
    bx1, by1, bx2, by2 = [int(v) for v in bbox]
    bw = bx2 - bx1
    bh = by2 - by1
    pad_x = int(bw * 0.15)
    pad_y = int(bh * 0.15)
    
    x1 = max(0, bx1 - pad_x)
    y1 = max(0, by1 - pad_y)
    x2 = min(w, bx2 + pad_x)
    y2 = min(h, by2 + pad_y)
    
    roi_gray = gray[y1:y2, x1:x2].copy()
    roi_mask = yolo_mask_u8[y1:y2, x1:x2]
    
    # 1. Ön İşleme (imageProcessing.py mantığı)
    blurred = cv2.bilateralFilter(roi_gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # Maske merkezini bul
    M = cv2.moments(roi_mask)
    if M['m00'] == 0:
        return None
    fcx = int(M['m10'] / M['m00'])
    fcy = int(M['m01'] / M['m00'])
    mask_area = cv2.countNonZero(roi_mask)
    
    # 2. Canny Edge Detection (Otomatik Eşik)
    v = np.median(enhanced)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(enhanced, lower, upper)
    
    # 3. Morfolojik Kapatma (Kırık çizgileri birleştir - imageProcessing.py gibi iter=1 veya 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # 4. Kontur Arama (RETR_TREE hiyerarşisi çok önemli)
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours or hierarchy is None:
        return None
        
    hierarchy = hierarchy[0]
    candidates = []
    
    rh, rw = roi_gray.shape
    
    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        # Area filtreleri çok geniş tutuldu (YOLO zaten sadece camı verdi)
        if area < mask_area * 0.05 or area > mask_area * 2.0:
            continue
            
        cx, cy, cw, ch = cv2.boundingRect(cnt)
        aspect = float(cw) / ch if ch > 0 else 0
        if not (0.3 < aspect < 4.0):
            continue
            
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / (hull_area + 1e-6)
        if solidity < 0.60: # Gevşetildi
            continue
            
        # Puanlama (imageProcessing.py mantığı)
        is_inner = hierarchy[i][3] != -1
        
        y_max = cnt[:, :, 1].max()
        bottomness = (y_max / rh) ** 2
        
        score = area * bottomness
        if is_inner:
            score *= 3.0
            
        candidates.append((score, cnt))
        
    if not candidates:
        # HİÇBİR ŞEY BULUNAMAZSA (Örn. çok saydam çerçeve): 
        # Canny çuvalladı demektir. O zaman doğrudan YOLO'nun maskesini kullan!
        mask_contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if mask_contours:
            best_cnt = max(mask_contours, key=cv2.contourArea)
        else:
            return None
    else:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_cnt = candidates[0][1]
    
    # Shape Completion (imageProcessing.py'deki gibi fitEllipse fallback convexHull)
    if len(best_cnt) >= 5:
        try:
            ell = cv2.fitEllipse(best_cnt)
            center_e = (int(ell[0][0]), int(ell[0][1]))
            axes_e = (int(ell[1][0] / 2), int(ell[1][1] / 2))
            angle_e = int(ell[2])
            hull_poly = cv2.ellipse2Poly(center_e, axes_e, angle_e, 0, 360, 5)
            final_contour = hull_poly.reshape(-1, 1, 2)
        except Exception:
            final_contour = cv2.convexHull(best_cnt)
    else:
        final_contour = cv2.convexHull(best_cnt)
        
    # Orijinal resim koordinatlarına kaydır
    final_contour = final_contour + [x1, y1]
    return final_contour

def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return
        
    results = yolo_model(img, conf=0.25, verbose=False)
    lens_contours = []
    
    if len(results) > 0:
        result = results[0]
        if result.masks is not None and result.boxes is not None:
            for mask, box in zip(result.masks.data, result.boxes):
                cls_id = int(box.cls[0].item())
                if cls_id == 1: # Sadece cam (glass)
                    bbox = box.xyxy[0].cpu().numpy()
                    
                    mask_np = mask.cpu().numpy()
                    mask_resized = cv2.resize(mask_np, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
                    mask_u8 = (mask_resized * 255).astype(np.uint8)
                    
                    # Gerçek Image Processing teknikleriyle ince kenar bul
                    refined_contour = refine_with_image_processing(img, mask_u8, bbox)
                    if refined_contour is not None:
                        lens_contours.append(refined_contour)

    if not lens_contours:
        print(f"[{os.path.basename(img_path)}] Cam tespit edilemedi.")
        return

    for cnt in lens_contours:
        cv2.drawContours(img, [cnt], -1, (255, 255, 0), 2, cv2.LINE_AA)

    out_name = os.path.basename(img_path).replace('.', '_precision.')
    out_path = os.path.join(OUTPUT_DIR, out_name)
    cv2.imwrite(out_path, img)
    print(f"[{os.path.basename(img_path)}] İşlendi ve kaydedildi -> {out_path}")

def main():
    image_exts = ("*.png", "*.jpg", "*.jpeg")
    image_paths = []
    for ext in image_exts:
        image_paths.extend(glob.glob(os.path.join(IMAGES_DIR, ext)))
        
    for p in image_paths[:5]:
        process_image(p)

if __name__ == "__main__":
    main()
