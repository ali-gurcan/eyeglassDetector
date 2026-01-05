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
import argparse
from pathlib import Path
import cv2
import numpy as np

# --- Configuration ---
INPUT_FOLDER = 'images'
OUTPUT_FOLDER = 'output'
DEBUG_FOLDER = 'debug'
FACE_CASCADE_PATH = 'haarcascades/haarcascade_frontalface_default.xml'
# CHANGE: Use specialized eyeglasses cascade
EYE_CASCADE_PATH = 'haarcascades/haarcascade_eye_tree_eyeglasses.xml'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(DEBUG_FOLDER, exist_ok=True)

# Global debug flag
DEBUG_MODE = False

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

def find_frame_candidate(roi_gray, sensitivity='normal', debug_prefix=''):
    """
    Core detection logic with variable sensitivity.
    Returns: (best_contour, debug_info_dict)
    """
    debug_info = {
        'total_contours': 0,
        'filtered_by_area_min': 0,
        'filtered_by_area_max': 0,
        'filtered_by_aspect': 0,
        'filtered_by_position': 0,
        'candidates': 0
    }
    
    h_roi, w_roi = roi_gray.shape
    roi_area = h_roi * w_roi

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
    edges = cv2.Canny(enhanced, lower, upper)
    
    # 3. Morfolojik Kapatma (Boşluk Doldurma)
    # Çerçeve kırıksa birleştir. Yatay kernel kullanıyoruz.
    k_size = (3, 3) if sensitivity == 'normal' else (5, 5) 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_size)
    # Iteration düşürüldü (2->1): Göz ile birleşmeyi önlemek için.
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # DEBUG: Ara görüntüleri kaydet
    if DEBUG_MODE and debug_prefix:
        debug_path = Path(DEBUG_FOLDER)
        cv2.imwrite(str(debug_path / f"{debug_prefix}_1_blurred.png"), blurred)
        cv2.imwrite(str(debug_path / f"{debug_prefix}_2_enhanced.png"), enhanced)
        cv2.imwrite(str(debug_path / f"{debug_prefix}_3_edges.png"), edges)
        cv2.imwrite(str(debug_path / f"{debug_prefix}_4_closed.png"), closed)
    
    # 4. Kontur Arama
    # RETR_TREE: İç içe geçmiş her şeyi bul
    contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    debug_info['total_contours'] = len(contours)
    
    candidates = []
    
    # DEBUG: Konturları görselleştir
    debug_vis = None
    if DEBUG_MODE and debug_prefix:
        debug_vis = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
    
    # Hierarchy kontrolü: [Next, Previous, First_Child, Parent]
    # Parent != -1 ise bu bir "iç kontur"dur (Frame'in deliği/camı olabilir)
    if hierarchy is not None:
        hierarchy = hierarchy[0]
    else:
        hierarchy = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = float(w) / h if h > 0 else 0
        
        # İç Kontur Kontrolü (Parent'ı var mı?)
        parent_idx = hierarchy[i][3] if len(hierarchy) > i else -1
        is_inner = (parent_idx != -1)
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # --- FİLTRELER ---
        # Normal modda sıkı, gevşek modda toleranslı filtreler
        # CHANGE: 0.05 -> 0.10 (Küçük göz parlamalarını/gürültüyü at)
        min_area_ratio = 0.10 if sensitivity == 'normal' else 0.05
        # CHANGE: 0.40 -> 0.30 (Çerçeve kalınsa elensin, sadece ince cam kalsın)
        max_area_ratio = 0.30 if sensitivity == 'normal' else 0.50
        min_aspect = 0.6 if sensitivity == 'normal' else 0.4
        # CHANGE: 4.0 -> 2.2 (Tek cam arıyoruz, 2.5'tan uzun olamaz)
        max_aspect = 2.2 if sensitivity == 'normal' else 3.0
        
        # CHANGE: 0.72 -> 0.85 (Daha pürüzsüz şekiller iste)
        min_solidity = 0.85 if sensitivity == 'normal' else 0.70
        
        # DEBUG: Filtre takılma nedenlerini kaydet
        if area < roi_area * min_area_ratio:
            debug_info['filtered_by_area_min'] += 1
            if DEBUG_MODE and debug_vis is not None:
                cv2.rectangle(debug_vis, (x, y), (x+w, y+h), (0, 0, 255), 1)  # Kırmızı: çok küçük
            continue
        if area > roi_area * max_area_ratio:
            debug_info['filtered_by_area_max'] += 1
            if DEBUG_MODE and debug_vis is not None:
                cv2.rectangle(debug_vis, (x, y), (x+w, y+h), (255, 0, 255), 1)  # Magenta: çok büyük
            continue
        if not (min_aspect < aspect < max_aspect):
            debug_info['filtered_by_aspect'] += 1
            if DEBUG_MODE and debug_vis is not None:
                cv2.rectangle(debug_vis, (x, y), (x+w, y+h), (255, 165, 0), 1)  # Turuncu: aspect ratio
            continue
        
        # Solidity Filtresi (Yamuk yumuk şekilleri at)
        if solidity < min_solidity:
             if DEBUG_MODE and debug_vis is not None:
                cv2.rectangle(debug_vis, (x, y), (x+w, y+h), (128, 0, 128), 1) # Mor: Bozuk şekil
             continue

        # Konum Filtresi (Kaş Kontrolü)
        # CHANGE: 0.05 -> 0.15 (ROI'nin tepesine çok yakınsa muhtemelen kaştır)
        if y < h_roi * 0.15:
            debug_info['filtered_by_position'] += 1
            if DEBUG_MODE and debug_vis is not None:
                cv2.rectangle(debug_vis, (x, y), (x+w, y+h), (0, 255, 255), 1)  # Cyan: konum
            continue
        
        # Adayı kaydet: (İç_Kontur_Mu, Alan, Kontur)
        # Öncelik: 1. İç Kontur olması, 2. Alan büyüklüğü
        candidates.append((is_inner, area, cnt))
        debug_info['candidates'] += 1
        if DEBUG_MODE and debug_vis is not None:
            color = (0, 255, 0) if is_inner else (0, 100, 0) # Parlak yeşil: iç, Koyu yeşil: dış
            cv2.rectangle(debug_vis, (x, y), (x+w, y+h), color, 2)
    
    # DEBUG: Görselleştirmeyi kaydet
    if DEBUG_MODE and debug_vis is not None and debug_prefix:
        cv2.imwrite(str(Path(DEBUG_FOLDER) / f"{debug_prefix}_5_contours_filtered.png"), debug_vis)
    
    # Sıralama Stratejisi:
    # 1. "İç Kontur" (Cam deliği) her zaman önceliklidir.
    # 2. Eğer iç kontur yoksa, boyutu "İdeal Gözlük Camı"na en yakın olanı seç.
    #    (En büyüğü seçersek genelde dış çerçeveyi alıyoruz, bu yanlış.)
    target_area = roi_area * 0.20 # İdeal cam boyutu tahminimiz (%20)
    
    if candidates:
        # x[0]: is_inner (Boole, True=1, False=0)
        # x[1]: area
        # Sıralama: is_inner (Büyükten küçüğe), sonra hedef alana yakınlık (Fark küçükten büyüğe -> -Fark büyükten küçüğe)
        candidates.sort(key=lambda x: (x[0], -abs(x[1] - target_area)), reverse=True)
        return candidates[0][2], debug_info  # En iyi konturu döndür
    
    return None, debug_info

def process_single_eye_roi(roi_gray, roi_color, offset_x, offset_y, debug_prefix=''):
    if roi_gray.size == 0: 
        return 0, {}

    # PLAN A: Normal Hassasiyet (Temiz görüntü arar)
    best_cnt, debug_info_normal = find_frame_candidate(roi_gray, sensitivity='normal', debug_prefix=f"{debug_prefix}_normal")
    
    # PLAN B: Eğer Plan A başarısızsa, "Aggressive Mode" aç (Daha toleranslı)
    debug_info_relaxed = {}
    if best_cnt is None:
        best_cnt, debug_info_relaxed = find_frame_candidate(roi_gray, sensitivity='relaxed', debug_prefix=f"{debug_prefix}_relaxed")
        
    if best_cnt is not None:
        # CHANGE: Convex Hull KAPALI. Doğrudan konturu çiz.
        # Böylece "uydurma" şekiller yerine ne bulduysak onu görürüz.
        contour_offset = best_cnt + [offset_x, offset_y]
        
        # Çizim (Cyan, Kalınlık 2)
        cv2.drawContours(roi_color, [contour_offset], -1, (255, 255, 0), 2)
        
        # DEBUG: Birleştirilmiş debug bilgisi
        combined_debug = {
            'mode_used': 'normal' if debug_info_relaxed.get('total_contours', 0) == 0 else 'relaxed',
            'normal': debug_info_normal,
            'relaxed': debug_info_relaxed
        }
        return 1, combined_debug
            
    # DEBUG: Başarısız durumda da bilgi döndür
    combined_debug = {
        'mode_used': 'failed',
        'normal': debug_info_normal,
        'relaxed': debug_info_relaxed
    }
    return 0, combined_debug

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
    all_debug_info = []
    
    for face_idx, (fx, fy, fw, fh) in enumerate(faces):
        # Göz Şeridi - Çerçevelerin alt kısmını da yakalamak için daha aşağıya genişletildi
        # CHANGE: 0.15 -> 0.20 (Kaşları daha iyi elemek için biraz daha indi)
        roi_y_start = int(fy + fh * 0.20)  
        roi_y_end = int(fy + fh * 0.60)    # CHANGE: 0.65 -> 0.60 (Çene kısmındaki gürültüyü azalt)
        roi_x_mid = int(fx + fw / 2)
        
        # Static Strip ROI (Sağlam Yöntem)
        coords = [
            (fx, roi_y_start, roi_x_mid, roi_y_end, fx, roi_y_start, 'left'),      # SOL
            (roi_x_mid, roi_y_start, fx+fw, roi_y_end, roi_x_mid, roi_y_start, 'right') # SAĞ
        ]

        for (x1, y1, x2, y2, off_x, off_y, side) in coords:
            y1=max(0, y1); y2=min(gray.shape[0], y2)
            x1=max(0, x1); x2=min(gray.shape[1], x2)
            
            roi_g = gray[y1:y2, x1:x2]
            
            # ROI çok küçükse atla (Hata önleyici)
            if roi_g.size == 0 or roi_g.shape[0] < 10 or roi_g.shape[1] < 10:
                continue
            
            # DEBUG: Prefix oluştur
            debug_prefix = f"{input_path.stem}_face{face_idx}_{side}" if DEBUG_MODE else ""
            
            hulls_found, debug_info = process_single_eye_roi(roi_g, img, off_x, off_y, debug_prefix)
            total_hulls += hulls_found
            
            if DEBUG_MODE:
                debug_info['side'] = side
                debug_info['face_idx'] = face_idx
                all_debug_info.append(debug_info)

    output_path = Path(OUTPUT_FOLDER) / f"{input_path.stem}_aggressive{input_path.suffix}"
    cv2.imwrite(str(output_path), img)
    
    if total_hulls > 0:
        print(f"[BASARILI] {filename} -> {total_hulls} çerçeve bulundu.")
    else:
        print(f"[UYARI] {filename} -> Tüm denemelere rağmen bulunamadı.")
    
    # DEBUG: Detaylı bilgi yazdır
    if DEBUG_MODE and all_debug_info:
        print(f"  [DEBUG] {filename} detayları:")
        for idx, dbg in enumerate(all_debug_info):
            mode = dbg.get('mode_used', 'unknown')
            normal = dbg.get('normal', {})
            relaxed = dbg.get('relaxed', {})
            side = dbg.get('side', 'unknown')
            
            print(f"    {side.upper()} göz (Face {dbg.get('face_idx', 0)}):")
            print(f"      Mod: {mode}")
            if normal.get('total_contours', 0) > 0:
                print(f"      Normal mod: {normal['total_contours']} kontur bulundu")
                print(f"        - Alan min filtresi: {normal['filtered_by_area_min']}")
                print(f"        - Alan max filtresi: {normal['filtered_by_area_max']}")
                print(f"        - Aspect ratio filtresi: {normal['filtered_by_aspect']}")
                print(f"        - Konum filtresi: {normal['filtered_by_position']}")
                print(f"        - Geçen adaylar: {normal['candidates']}")
            if relaxed.get('total_contours', 0) > 0:
                print(f"      Relaxed mod: {relaxed['total_contours']} kontur bulundu")
                print(f"        - Alan min filtresi: {relaxed['filtered_by_area_min']}")
                print(f"        - Alan max filtresi: {relaxed['filtered_by_area_max']}")
                print(f"        - Aspect ratio filtresi: {relaxed['filtered_by_aspect']}")
                print(f"        - Konum filtresi: {relaxed['filtered_by_position']}")
                print(f"        - Geçen adaylar: {relaxed['candidates']}")

def main():
    global DEBUG_MODE
    
    parser = argparse.ArgumentParser(description='Aggressive Eyeglass Detection Pipeline')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug mode: save intermediate images and detailed logs')
    args = parser.parse_args()
    
    DEBUG_MODE = args.debug
    
    if DEBUG_MODE:
        print("--- Starting AGGRESSIVE DETECTION Pipeline (DEBUG MODE) ---")
        print(f"[DEBUG] Debug görüntüleri '{DEBUG_FOLDER}/' klasörüne kaydedilecek.")
    else:
        print("--- Starting AGGRESSIVE DETECTION Pipeline ---")
    
    try:
        face_cascade = load_face_cascade()
    except Exception as e:
        print(f"[FATAL] {e}")
        return
    
    clear_output_dir()
    
    # DEBUG klasörünü de temizle
    if DEBUG_MODE:
        debug_path = Path(DEBUG_FOLDER)
        if debug_path.exists():
            for item in debug_path.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                except OSError:
                    pass
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
        image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    
    if not image_files:
        print(f"[WARN] '{INPUT_FOLDER}/' klasöründe resim bulunamadı.")
        return
    
    print(f"Toplam {len(image_files)} resim işlenecek...\n")
    
    for img_path in image_files:
        process_image(img_path, face_cascade)
    
    print("\n--- Bitti ---")
    if DEBUG_MODE:
        print(f"[DEBUG] Detaylı görüntüler '{DEBUG_FOLDER}/' klasöründe.")

if __name__ == "__main__":
    main()