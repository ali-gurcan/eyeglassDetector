#!/usr/bin/env python3
"""
Aggressive Eyeglass Detection (No-Fail Edition)
----------------------------------------
Priorities:
1. Detection Rate > Aesthetics. (Find it even if it's messy).
2. Heavy Morphological Closing to connect broken frames.
3. Retry Mechanism: If strict parameters fail, retry with relaxed ones.
4. RETR_TREE: Look for inner frame boundaries if outer ones are lost.

Author: Senior CV Engineer (AI Assistant)
"""

import glob
import os
import shutil
from pathlib import Path
import cv2
import numpy as np

# --- Configuration ---
INPUT_FOLDER = 'images'
OUTPUT_FOLDER = 'output'
FACE_CASCADE_PATH = 'haarcascades/haarcascade_frontalface_default.xml'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

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

def load_face_cascade():
    face_path = Path(FACE_CASCADE_PATH)
    if not face_path.exists():
        face_path = Path(cv2.data.haarcascades) / 'haarcascade_frontalface_default.xml'
    return cv2.CascadeClassifier(str(face_path))

def find_frame_candidate(roi_gray, sensitivity='normal'):
    """
    Core detection logic with variable sensitivity.
    """
    # 1. Ön İşleme
    # Bilateral filtre kenarları korur ama gürültüyü atar
    blurred = cv2.bilateralFilter(roi_gray, 9, 75, 75)
    
    # CLAHE ile kontrastı patlat (Koyu çerçeveleri ortaya çıkar)
    clip_limit = 2.0 if sensitivity == 'normal' else 4.0
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    
    # 2. Kenar Tespiti (Canny)
    # Hassas modda eşikleri düşür
    v = np.median(enhanced)
    sigma = 0.33 if sensitivity == 'normal' else 0.50 # Daha geniş aralık
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(enhanced, lower, upper)
    
    # 3. Morfolojik Kapatma (Boşluk Doldurma)
    # Çerçeve kırıksa birleştir. Yatay kernel kullanıyoruz.
    k_size = (3, 3) if sensitivity == 'normal' else (5, 5) # Hassas modda daha kalın birleştir
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_size)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 4. Kontur Arama
    # RETR_TREE: İç içe geçmiş her şeyi bul
    contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    h_roi, w_roi = roi_gray.shape
    roi_area = h_roi * w_roi
    
    candidates = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = float(w) / h
        
        # --- FİLTRELER ---
        # Normal modda sıkı, gevşek modda toleranslı filtreler
        min_area_ratio = 0.05 if sensitivity == 'normal' else 0.02
        max_area_ratio = 0.60 if sensitivity == 'normal' else 0.80
        min_aspect = 0.8 if sensitivity == 'normal' else 0.5
        max_aspect = 4.0 if sensitivity == 'normal' else 6.0
        
        if area < roi_area * min_area_ratio: continue
        if area > roi_area * max_area_ratio: continue
        if not (min_aspect < aspect < max_aspect): continue

        # Konum Filtresi (Kaş Kontrolü)
        # Sadece çok bariz kaşları at (en tepedekiler)
        if y < h_roi * 0.05: continue 
        
        # Adayı kaydet (Alan büyüklüğüne göre)
        candidates.append((area, cnt))
        
    # En büyük alanı döndür
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1] # En iyi konturu döndür
    
    return None

def process_single_eye_roi(roi_gray, roi_color, offset_x, offset_y):
    if roi_gray.size == 0: return 0

    # PLAN A: Normal Hassasiyet (Temiz görüntü arar)
    best_cnt = find_frame_candidate(roi_gray, sensitivity='normal')
    
    # PLAN B: Eğer Plan A başarısızsa, "Aggressive Mode" aç (Daha toleranslı)
    if best_cnt is None:
        best_cnt = find_frame_candidate(roi_gray, sensitivity='relaxed')
        
    if best_cnt is not None:
        # Şekli düzgünleştirmek için Convex Hull (Lastik Bant) kullan
        # Bu, kırık parçaları birleştirip bütün bir çerçeve gibi gösterir
        hull = cv2.convexHull(best_cnt)
        
        contour_offset = hull + [offset_x, offset_y]
        
        # Çizim (Cyan, Kalınlık 2)
        cv2.drawContours(roi_color, [contour_offset], -1, (255, 255, 0), 2)
        return 1
            
    return 0

def process_image(img_path, face_cascade):
    input_path = Path(img_path)
    filename = input_path.name
    img = cv2.imread(str(input_path))
    
    if img is None: return

    # Standart boyutlandırma
    if img.shape[1] > 1000:
        scale = 1000 / img.shape[1]
        img = cv2.resize(img, None, fx=scale, fy=scale)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Yüz Tespiti
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    # Fallback: Yüz bulunamazsa
    if len(faces) == 0:
        h, w = img.shape[:2]
        faces = [[int(w*0.25), int(h*0.2), int(w*0.5), int(h*0.6)]]

    # En büyük yüz
    if len(faces) > 0:
         faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    total_hulls = 0
    for (fx, fy, fw, fh) in faces:
        # Göz Şeridi
        roi_y_start = int(fy + fh * 0.18) 
        roi_y_end = int(fy + fh * 0.55)
        roi_x_mid = int(fx + fw / 2)
        
        coords = [
            (fx, roi_y_start, roi_x_mid, roi_y_end, fx, roi_y_start),      # SOL
            (roi_x_mid, roi_y_start, fx+fw, roi_y_end, roi_x_mid, roi_y_start) # SAĞ
        ]

        for (x1, y1, x2, y2, off_x, off_y) in coords:
            y1=max(0, y1); y2=min(gray.shape[0], y2)
            x1=max(0, x1); x2=min(gray.shape[1], x2)
            
            roi_g = gray[y1:y2, x1:x2]
            
            # ROI çok küçükse atla (Hata önleyici)
            if roi_g.size == 0 or roi_g.shape[0] < 10 or roi_g.shape[1] < 10:
                continue
                
            total_hulls += process_single_eye_roi(roi_g, img, off_x, off_y)

    output_path = Path(OUTPUT_FOLDER) / f"{input_path.stem}_aggressive{input_path.suffix}"
    cv2.imwrite(str(output_path), img)
    
    if total_hulls > 0:
        print(f"[BASARILI] {filename} -> {total_hulls} çerçeve bulundu.")
    else:
        print(f"[UYARI] {filename} -> Tüm denemelere rağmen bulunamadı.")

def main():
    print("--- Starting AGGRESSIVE DETECTION Pipeline ---")
    try:
        face_cascade = load_face_cascade()
    except Exception:
        return
    clear_output_dir()
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
        image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    
    for img_path in image_files:
        process_image(img_path, face_cascade)
    print("--- Bitti ---")

if __name__ == "__main__":
    main()