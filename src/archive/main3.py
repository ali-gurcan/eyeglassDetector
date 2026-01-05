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

def load_eye_cascade():
    eye_path = Path(EYE_CASCADE_PATH)
    if not eye_path.exists():
        # Fallback to standard eye cascade if specialized one missing
        eye_path = Path(cv2.data.haarcascades) / 'haarcascade_eye_tree_eyeglasses.xml'
    return cv2.CascadeClassifier(str(eye_path))

def calculate_sclera_contrast(roi_gray, pupil_center, pupil_radius=5):
    """
    Calculates a score based on the contrast between the pupil (dark) and the surrounding sclera (bright).
    Enforces "Horizontal White" rule: Sclera must be visible on Left AND Right.
    """
    h, w = roi_gray.shape
    cx, cy = pupil_center
    r = max(2, int(pupil_radius))
    
    # 1. Pupil Mask (Inner)
    mask_pupil = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask_pupil, (cx, cy), r, 255, -1)
    
    # 2. Horizontal Flank Masks (Left & Right)
    mask_left = np.zeros((h, w), dtype=np.uint8)
    mask_right = np.zeros((h, w), dtype=np.uint8)
    
    y1, y2 = max(0, cy - r), min(h, cy + r)
    
    # Left Box: [cx-3r, cx-r]
    lx1, lx2 = max(0, cx - 3*r), max(0, cx - r)
    if lx2 > lx1: cv2.rectangle(mask_left, (lx1, y1), (lx2, y2), 255, -1)
        
    # Right Box: [cx+r, cx+3r]
    rx1, rx2 = min(w, cx + r), min(w, cx + 3*r)
    if rx2 > rx1: cv2.rectangle(mask_right, (rx1, y1), (rx2, y2), 255, -1)
    
    # Check emptiness (Edges/Corners)
    if cv2.countNonZero(mask_left) == 0 or cv2.countNonZero(mask_right) == 0:
        return -1000 # Reject: No horizontal space
        
    mean_pupil = cv2.mean(roi_gray, mask=mask_pupil)[0]
    mean_left = cv2.mean(roi_gray, mask=mask_left)[0]
    mean_right = cv2.mean(roi_gray, mask=mask_right)[0]
    
    # DIRECTIONAL VALIDATION (User: "Left/Right must be white")
    # Both sides must be brighter than pupil + margin
    margin = 5
    if mean_left < mean_pupil + margin or mean_right < mean_pupil + margin:
        return -500 # Severe Penalty: Not horizontally flanked by white
        
    # Average Sclera Brightness for scoring
    mean_sclera = (mean_left + mean_right) / 2
    contrast = mean_sclera - mean_pupil
    
    # 3. ANTI-GLARE VETO (User: "Selected white area")
    # A pupil CANNOT be bright or brighter than sclera
    if mean_pupil > 110: # Absolute brightness verification (Gray value > 110 is too bright for a pupil)
        return -1000
    if mean_pupil > mean_sclera: # It must be darker than surroundings
        return -1000
    if contrast < 10: # Must have some minimal contrast
        return -1000
    
    # CHANGE: DOMINANT SIZE SCORING (User: "Big black dot score should be very high")
    # New Formula: Radius * 15 (Max 300)
    # This ensures size overpowers contrast (~50) and center penalty (~30)
    area_score = min(300, pupil_radius * 15)
    
    # CHANGE: Central Bias Penalty (User Request: "Search in middle")
    center_x, center_y = w // 2, h // 2
    dist_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
    penalty = dist_from_center * 0.5 # Penalty factor
    
    return contrast + area_score - penalty

def find_frame_candidate(roi_gray, roi_color_crop, eye_cascade=None, sensitivity='normal', debug_prefix='', morph_iters=1, use_mask=False):
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
    blurred = cv2.bilateralFilter(roi_gray, 9, 75, 75)
    
    clip_limit = 2.0 if sensitivity == 'normal' else 4.0
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    
    # -- STEP 0: PUPIL DETECTION (Her şeyden önce) --
    # Göz merkezini bul (Always run this)
    # CHANGE: Tightened margins to 0.30 (Central 40% only)
    margin_h = int(h_roi * 0.30)
    margin_w = int(w_roi * 0.30)
    center_roi = enhanced[margin_h:-margin_h, margin_w:-margin_w]
    # Extract color ROI as well
    if roi_color_crop is not None:
         center_roi_color = roi_color_crop[margin_h:-margin_h, margin_w:-margin_w]
    else:
         center_roi_color = None
    
    pupil_x, pupil_y = w_roi // 2, h_roi // 2 # Default center
    if center_roi.size > 0:
        # ROBUST PUPIL DETECTION (Multi-Stage Adaptive)
        # Priority 0: STRICT COLOR CHECK (RGB < 20)
        # User Rule: "rgb değerlerinin üçü de 20nin altındadır tam siyah olmasa bile"
        found_pupil = False
        best_pupil_candidate = None
        best_pupil_score = -999
        
        if center_roi_color is not None:
            # Create mask for pixels where R<20 AND G<20 AND B<20
            # Note: OpenCV uses BGR
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([20, 20, 20])
            mask_black = cv2.inRange(center_roi_color, lower_black, upper_black)
            
            # Find blobs in this strict mask
            cnts_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if cnts_black:
                # Find the largest valid blob
                # CHANGE: Increased min_area to 75 (User: "Slightly Larger Black Dot")
                valid_black_blobs = [c for c in cnts_black if cv2.contourArea(c) > 75] 
                if valid_black_blobs:
                    largest_black = max(valid_black_blobs, key=cv2.contourArea)
                    M = cv2.moments(largest_black)
                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        best_pupil_candidate = (cx, cy)
                        pupil_x = cx + margin_w
                        pupil_y = cy + margin_h
                        found_pupil = True
                        best_pupil_score = 10000

        # Priority 0.5: HAAR CASCADE (Standard Robustness)
        # If strict color failed, trust the cascade if provided
        if not found_pupil and eye_cascade is not None:
             # Run Cascade on the ENHANCED ROI (or blurred)
             # Cascade expects standard contrast
             # Important: Cascade runs on the whole ROI, but returns relative coords
             eyes = eye_cascade.detectMultiScale(center_roi, 1.1, 3, minSize=(15, 15))
             
             if len(eyes) > 0:
                 # Find the largest eye (assumption: ROI mostly contains one eye)
                 ex, ey, ew, eh = sorted(eyes, key=lambda e: e[2]*e[3], reverse=True)[0]
                 # Set candidate relative to center_roi (consistent with other methods)
                 best_pupil_candidate = (ex + ew // 2, ey + eh // 2)
                 found_pupil = True
                 best_pupil_score = 5000 # High priority but less than strict color
        
        # Priority 1-3: Adaptive Search (Hybrid/Fallback)
        if not found_pupil:
             # Search ONLY in center_roi to avoid eyebrows
             # CHANGE: Increased min_area thresholds
             adaptive_params = [
                 {'name': 'static_strict', 'thresh_pct': 15, 'min_circ': 0.5, 'min_area': 70},
                 {'name': 'static_relaxed', 'thresh_pct': 25, 'min_circ': 0.3, 'min_area': 50},
                 {'name': 'static_desperate', 'thresh_pct': 40, 'min_circ': 0.15, 'min_area': 30}
             ]
             
             roi_blurred_pupil = cv2.GaussianBlur(center_roi, (7, 7), 0)
             
             for stage in adaptive_params:
                 candidates = []
            
                 # METHOD A: HOUGH CIRCLES (Gradient)
                 # Only in earlier stages to avoid false positives
                 if stage['name'] != 'static_desperate':
                     try:
                         circles = cv2.HoughCircles(
                             roi_blurred_pupil, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                             param1=50, param2=30, minRadius=5, maxRadius=25
                         )
                         if circles is not None:
                             circles = np.uint16(np.around(circles))
                             for i in circles[0, :]:
                                 cx_h, cy_h, r_h = i[0], i[1], i[2]
                                 # Create a dummy contour for consistency
                                 # Approximation of circle as polygon
                                 h_pts = cv2.ellipse2Poly((int(cx_h), int(cy_h)), (int(r_h), int(r_h)), 0, 0, 360, 10)
                                 h_cnt = h_pts.reshape(-1, 1, 2)
                                 candidates.append(h_cnt)
                     except:
                         pass

                 # METHOD B: BLOB DETECTION (Intensity)
                 min_val, _, _, _ = cv2.minMaxLoc(roi_blurred_pupil)
                 # Dynamic thresholding based on darkness percentile
                 thresh_val = min_val + (255 - min_val) * (stage['thresh_pct'] / 100.0)
                 _, binary_map = cv2.threshold(roi_blurred_pupil, thresh_val, 255, cv2.THRESH_BINARY_INV)
                 
                 blobs, _ = cv2.findContours(binary_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                 for b in blobs:
                     area = cv2.contourArea(b)
                     if area < stage['min_area']: continue
                     
                     # Circularity Check
                     perimeter = cv2.arcLength(b, True)
                     if perimeter == 0: continue
                     circularity = 4 * np.pi * (area / (perimeter * perimeter))
                     
                     if circularity > stage['min_circ']:
                         candidates.append(b)
                 
                 # EVALUATE CANDIDATES (Sclera Contrast Score)
                 for cand in candidates:
                     M = cv2.moments(cand)
                     if M["m00"] == 0: continue
                     cx = int(M["m10"] / M["m00"])
                     cy = int(M["m01"] / M["m00"])
                     
                     # Radius approximation
                     _, radius = cv2.minEnclosingCircle(cand)
                     
                     score = calculate_sclera_contrast(roi_blurred_pupil, (cx, cy), pupil_radius=int(radius))
                     
                     if score > best_pupil_score:
                         best_pupil_score = score
                         best_pupil_candidate = (cx, cy)
                         found_pupil = True
                 
                 # If we found a good candidate in this stage, stop
                 if found_pupil and best_pupil_score > 30: # 30 is a decent contrast threshold
                     break
        
        if found_pupil:
             pupil_x = best_pupil_candidate[0] + margin_w
             pupil_y = best_pupil_candidate[1] + margin_h
        else:
             # Fallback: Klasik minMaxLoc
             _, _, min_loc_simple, _ = cv2.minMaxLoc(roi_blurred_pupil)
             pupil_x = min_loc_simple[0] + margin_w
             pupil_y = min_loc_simple[1] + margin_h
             
    if DEBUG_MODE and debug_prefix:
        # DEBUG: Göz tespitinin sonucunu hemen göster
        debug_path = Path("debug")
        debug_path.mkdir(exist_ok=True)
        pupil_vis = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        cv2.circle(pupil_vis, (pupil_x, pupil_y), 5, (0, 0, 255), -1)
        # Sclera contrast değerini yaz
        if found_pupil:
             cv2.putText(pupil_vis, f"Score: {best_pupil_score:.1f}", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(str(debug_path / f"{debug_prefix}_01_pupil_loc.png"), pupil_vis)

    if use_mask:
        # DYNAMIC PUPIL BLURRING (Akıllı Göz Bulanıklaştırma)
        # Bulduğumuz pupil noktasına göre devasa alanı blurla.
        
        # 1. Yumuşak Maske Oluştur
        # CHANGE: Alan son kez ayarlandı (%18 Width, %16 Height)
        # User Feedback: "genişlik 18 yükseklik 16 olsun"
        axes = (int(w_roi * 0.18), int(h_roi * 0.16))
        
        mask = np.zeros_like(enhanced, dtype=np.float32)
        cv2.ellipse(mask, (pupil_x, pupil_y), axes, 0, 0, 360, 1.0, -1)
        
        # Maskenin kenarlarını yumuşat (Hard Edge olmasın)
        mask = cv2.GaussianBlur(mask, (21, 21), 11)
        
        # 2. Görüntüyü Ağır Bulanıklaştır (Doku kaybı)
        heavy_blur = cv2.GaussianBlur(enhanced, (99, 99), 30)
        
        # 3. Blend (Karıştır)
        enhanced_blended = (enhanced * (1.0 - mask) + heavy_blur * mask).astype(np.uint8)
        
        if DEBUG_MODE and debug_prefix:
            # VISUALIZATION: Nereyi banladığımızı açıkça göster
            ban_vis = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            cv2.ellipse(ban_vis, (pupil_x, pupil_y), axes, 0, 0, 360, (0, 255, 255), 2) # Cyan Elips
            cv2.circle(ban_vis, (pupil_x, pupil_y), 5, (0, 0, 255), -1) # Kırmızı Nokta
            cv2.imwrite(str(debug_path / f"{debug_prefix}_02_eye_ban_vis.png"), ban_vis)
            
            # Bulanıklaştırılmış "Nuked" halini kaydet
            cv2.imwrite(str(debug_path / f"{debug_prefix}_03_nuked_eye.png"), enhanced_blended)
        
        enhanced = enhanced_blended

    # 2. Kenar Tespiti
    v = np.median(enhanced)
    sigma = 0.33 if sensitivity == 'normal' else 0.50
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(enhanced, lower, upper)
    
    # 3. Morfolojik Kapatma (Boşluk Doldurma)
    # Çerçeve kırıksa birleştir. Yatay kernel kullanıyoruz.
    k_size = (3, 3) if sensitivity == 'normal' else (5, 5) 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_size)
    # Iteration parametrik yapıldı (Smart Retry için)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=morph_iters)
    
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
    
    # Missing variable fix:
    roi_center = (w_roi // 2, h_roi // 2)
    
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
        
        # Filtre 3: Konum (Çok kenarda mı?)
        cx, cy = x + w//2, y + h//2
        dist_from_center = np.sqrt((cx - roi_center[0])**2 + (cy - roi_center[1])**2)
        max_dist = w_roi * 0.45
        if dist_from_center > max_dist:
            debug_info['filtered_by_position'] += 1
            continue
            
        # 6. SHAPE COMPLETION (Şekil Tamamlama)
        # User feedback: "Üstünden çizgi çiziyorsun"
        # ConvexHull "U" şeklindeki bir çerçeveyi düz çizgiyle kapatır ve gözü keser.
        # FitEllipse ise eğimi takip ederek gözün ETRAFINDAN dolaşan bir yay çizer.
        if len(cnt) >= 5:
            try:
                # Elips uydur ve çokgene çevir (360 derece kapalı)
                ellipse_fit = cv2.fitEllipse(cnt)
                center = (int(ellipse_fit[0][0]), int(ellipse_fit[0][1]))
                axes = (int(ellipse_fit[1][0] / 2), int(ellipse_fit[1][1] / 2))
                angle = int(ellipse_fit[2])
                
                # Elipsi çizgi noktalarına dök (5 derece hassasiyette)
                hull_pts = cv2.ellipse2Poly(center, axes, angle, 0, 360, 5)
                hull = hull_pts.reshape(-1, 1, 2) # Contour formatı
            except:
                # Hata olursa (aşırı düz çizgi vb.) fallback
                hull = cv2.convexHull(cnt)
        else:
            hull = cv2.convexHull(cnt)
        # BOTTOM-ANCHOR SCORING (Alt Odaklı Puanlama)
        # User: "Çizgi çekmeye alttan başla... o kısmı birleştir"
        # Mantık: Çerçevenin alt kenarı (yanak tarafı) en temiz bölgedir.
        # Puan = Alan * (Alt Konum Ağırlığı)
        y_max = cnt[:, :, 1].max()
        bottomness = (y_max / h_roi) ** 2  # Karesel artış ile alt kısmı ödüllendir
        
        score = area * bottomness
        
        # INNER CONTOUR PRIORITY (İç Kontur Önceliği)
        # User Feedback: "İçi cam kenarı, dışı çerçeve kenarı... dışını hull ediyorsun"
        # Çözüm: Eğer bu kontur bir "İç Kontur" ise (Hierarchy parent'ı varsa), 
        # puanını radikal şekilde artır. Çünkü aradığımız şey tam olarak bu deliktir.
        if is_inner:
            score *= 3.0 # Outer (Dış) konturu kesinlikle geçmeli
            
        candidates.append((score, cnt))
        debug_info['candidates'] += 1
        if DEBUG_MODE and debug_vis is not None:
            color = (0, 255, 0) if is_inner else (0, 100, 0) # Parlak yeşil: iç, Koyu yeşil: dış
            cv2.rectangle(debug_vis, (x, y), (x+w, y+h), color, 2)
    
    # DEBUG: Görselleştirmeyi kaydet
    if DEBUG_MODE and debug_vis is not None and debug_prefix:
        cv2.imwrite(str(Path(DEBUG_FOLDER) / f"{debug_prefix}_5_contours_filtered.png"), debug_vis)
    
    if not candidates:
        return None, debug_info

    # Puana göre sırala (En yüksek puan en üstte)
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # En iyi adayı al
    best_cnt = candidates[0][1]
    
    # User: "Kopsa bile birleştir" -> Convex Hull uygula
    hull = cv2.convexHull(best_cnt)
    
    if DEBUG_MODE and debug_prefix:
        debug_path = Path("debug")
        vis_hull = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(vis_hull, [hull], -1, (0, 0, 255), 2)
        cv2.imwrite(str(debug_path / f"{debug_prefix}_04_hull.png"), vis_hull)
    
    # TIGHTENING (Daraltma/Kenar Düzeltme)
    # User: "Biraz çerçeveye taşıyorsun, cam kenarı lazım"
    # Sebep: Morph_iters=3 işlemi kenarları şişirdi (Dilate etkisi).
    # Çözüm: Şimdi aynı miktarda Erode (Aşındırma) yaparak gerçeğe dön.
    if morph_iters > 0:
        mask_refine = np.zeros((h_roi, w_roi), dtype=np.uint8)
        cv2.drawContours(mask_refine, [hull], -1, 255, -1) # İçini doldur
        
        # Kapatma işleminde kullandığımız kernel ve iterasyon kadar geri al
        # Biraz daha agresif aşındırabiliriz çünkü Hull da dışbükey yapıp şişirdi.
        # k_size yukarıda tanımlı (3x3 veya 5x5)
        # Kapatma işleminde kullandığımız kernel ve iterasyon kadar geri al
        # Biraz daha agresif aşındırabiliriz çünkü Hull da dışbükey yapıp şişirdi.
        # k_size yukarıda tanımlı (3x3 veya 5x5)
        erode_iters = max(1, morph_iters // 2)
        mask_eroded = cv2.erode(mask_refine, kernel, iterations=erode_iters)
        
        refined_cnts, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if refined_cnts:
            # En büyüğünü al (Erozyon sonucu bölünebilir, ana parçayı istiyoruz)
            refined_hull = max(refined_cnts, key=cv2.contourArea)
            
            if DEBUG_MODE and debug_prefix:
                vis_tight = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(vis_tight, [refined_hull], -1, (0, 255, 0), 2)
                cv2.imwrite(str(debug_path / f"{debug_prefix}_05_final_tightened.png"), vis_tight)

            return refined_hull, debug_info
            
    return hull, debug_info

def process_single_eye_roi(roi_gray, roi_color_crop, eye_cascade, full_image, offset_x, offset_y, debug_prefix='', forced_iters=None):
    if roi_gray.size == 0: 
        return 0, {}, 0

    # Determine starting iterations
    # If forced_iters is provided (by Symmetry Check), use it.
    # Otherwise default to 1 (Safe mode).
    start_iters = forced_iters if forced_iters is not None else 1

    # PLAN A: Normal Hassasiyet
    # Eğer forced_iters varsa (Symmetry Retry), doğrudan Relaxed modda çalış.
    # Çünkü iter=2 kalın çerçeve yapar, normal modun filtrelerine takılır.
    plan_a_sensitivity = 'relaxed' if forced_iters else 'normal'
    
    best_cnt, debug_info_normal = find_frame_candidate(
        roi_gray, roi_color_crop=roi_color_crop, eye_cascade=eye_cascade, sensitivity=plan_a_sensitivity, debug_prefix=f"{debug_prefix}_normal", 
        morph_iters=start_iters
    )
    
    # 2. SMART RETRY MANTIĞI
    # Durum 1: Hiçbir şey bulunamadı.
    # Durum 2: Bulunan şey çok küçük (Muhtemel Göz Bebeği/İris).
    h_roi, w_roi = roi_gray.shape
    roi_area = h_roi * w_roi
    
    should_retry = False
    if best_cnt is None:
        should_retry = True
        reason = "not_found"
    else:
        area = cv2.contourArea(best_cnt)
        if area < roi_area * 0.15: # %15'ten küçükse şüpheli
            should_retry = True
            reason = f"too_small_{area:.0f}"

    # Eğer zaten forced_iters ile geldiysek (Symmetry Retry), tekrar retry yapma (sonsuz döngü olmasın)
    if forced_iters is not None:
        should_retry = False

    debug_info_retry = {}
    if should_retry:
        # Daha güçlü birleştirme ile tekrar dene (morph_iters + 2 -> Toplam 3)
        # Balyoz Yöntemi: 13.png gibi çok kopuk çerçeveleri zorla birleştir.
        # CHANGE: Dynamic Maske AKTİF.
        retry_iters = start_iters + 2
        retry_cnt, debug_info_retry = find_frame_candidate(
            roi_gray, roi_color_crop=roi_color_crop, eye_cascade=eye_cascade, sensitivity='relaxed', debug_prefix=f"{debug_prefix}_retry", 
            morph_iters=retry_iters, use_mask=True
        )
        
        if retry_cnt is not None:
             retry_area = cv2.contourArea(retry_cnt)
             if best_cnt is None or retry_area > cv2.contourArea(best_cnt):
                 best_cnt = retry_cnt
                 debug_info_normal['smart_retry'] = f"active_{reason}"

    # PLAN B: Relaxed Mode
    debug_info_relaxed = {}
    if best_cnt is None:
        best_cnt, debug_info_relaxed = find_frame_candidate(
            roi_gray, roi_color_crop=roi_color_crop, eye_cascade=eye_cascade, sensitivity='relaxed', debug_prefix=f"{debug_prefix}_relaxed", morph_iters=start_iters + 1
        )
        
    found_area = 0
    if best_cnt is not None:
        found_area = cv2.contourArea(best_cnt)
        
        # CHANGE: Convex Hull KAPALI. Doğrudan konturu çiz.
        contour_offset = best_cnt + [offset_x, offset_y]
        
        # Çizim (Cyan, Kalınlık 2) -> Draw on full_image
        cv2.drawContours(full_image, [contour_offset], -1, (255, 255, 0), 2)
        
        combined_debug = {
            'mode_used': 'normal' if debug_info_relaxed.get('total_contours', 0) == 0 else 'relaxed',
            'normal': debug_info_normal,
            'retry': debug_info_retry,
            'relaxed': debug_info_relaxed
        }
        return 1, combined_debug, found_area
            
    combined_debug = {
        'mode_used': 'failed',
        'normal': debug_info_normal,
        'relaxed': debug_info_relaxed
    }
    return 0, combined_debug, 0

def process_image(img_path, face_cascade, eye_cascade):
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
    
    # Results container: [face_idx][side] = (hulls_found, debug_info, area, roi_args)
    # roi_args: (roi_g, img, off_x, off_y, debug_prefix)
    eye_results = {}
    
    for face_idx, (fx, fy, fw, fh) in enumerate(faces):
        if face_idx not in eye_results: eye_results[face_idx] = {}
        
        # Göz Şeridi
        roi_y_start = int(fy + fh * 0.20)  
        roi_y_end = int(fy + fh * 0.60)
        roi_x_mid = int(fx + fw / 2)
        
        # Static Strip ROI
        coords = [
            (fx, roi_y_start, roi_x_mid, roi_y_end, fx, roi_y_start, 'left'),
            (roi_x_mid, roi_y_start, fx+fw, roi_y_end, roi_x_mid, roi_y_start, 'right')
        ]

        for (x1, y1, x2, y2, off_x, off_y, side) in coords:
            y1=max(0, y1); y2=min(gray.shape[0], y2)
            x1=max(0, x1); x2=min(gray.shape[1], x2)
            
            roi_g = gray[y1:y2, x1:x2]
            # NEW: Extract Color ROI for RGB Check
            roi_c = img[y1:y2, x1:x2]
            
            if roi_g.size == 0 or roi_g.shape[0] < 10 or roi_g.shape[1] < 10:
                continue
            
            debug_prefix = f"{input_path.stem}_face{face_idx}_{side}" if DEBUG_MODE else ""
            
            # 1. INITIAL PASS
            hulls_found, debug_info, area = process_single_eye_roi(roi_g, roi_c, eye_cascade, img, off_x, off_y, debug_prefix)
            
            # Store everything needed for potential retry
            eye_args = (roi_g, roi_c, eye_cascade, img, off_x, off_y, debug_prefix)
            eye_results[face_idx][side] = {
                'found': hulls_found,
                'debug': debug_info,
                'area': area,
                'args': eye_args
            }
            
    # 2. SYMMETRY CHECK & RETRY
    for face_idx, sides in eye_results.items():
        if 'left' in sides and 'right' in sides:
            l_res = sides['left']
            r_res = sides['right']
            
            l_area = l_res['area']
            r_area = r_res['area']
            
            # Eğer biri diğerinden çok daha küçükse (ör: %10 daha küçük)
            # User Hint: "Muhtemelen göz çevresidir, denemeye devam et"
            max_area = max(l_area, r_area)
            if max_area > 0:
                diff_ratio = abs(l_area - r_area) / max_area
                if diff_ratio > 0.15: # Symmetry threshold
                    target_side = 'left' if l_area < r_area else 'right'
                    target_res = sides[target_side]
                    
                    if DEBUG_MODE:
                        print(f"      [SYMMETRY] {filename} Face {face_idx}: {target_side.upper()} (%{diff_ratio*100:.1f} smaller) -> Retrying Force Iters=2...")
                    
                    # RETRY with FORCED ITERATIONS (Aggressive Join)
                    args = target_res['args']
                    new_found, new_debug, new_area = process_single_eye_roi(*args, forced_iters=2)
                    
                    # Update if improved (or at least valid)
                    sides[target_side]['found'] = new_found
                    sides[target_side]['debug'] = new_debug
                    sides[target_side]['area'] = new_area
                    sides[target_side]['debug']['symmetry_retry'] = True

    # 3. CONSOLIDATE RESULTS
    for face_idx, sides in eye_results.items():
        for side, res in sides.items():
            total_hulls += res['found']
            debug_info = res['debug']
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
            retry = dbg.get('retry', {})
            relaxed = dbg.get('relaxed', {})
            side = dbg.get('side', 'unknown')
            sym = dbg.get('symmetry_retry', False)
            
            print(f"    {side.upper()} göz (Face {dbg.get('face_idx', 0)}):")
            print(f"      Mod: {mode} {'(SYMMETRY RETRY)' if sym else ''}")
            if normal.get('total_contours', 0) > 0:
                print(f"      Normal mod: {normal['total_contours']} kontur bulundu. Smart Retry: {normal.get('smart_retry', 'inactive')}")
            if retry.get('total_contours', 0) > 0:
                print(f"      Smart Retry mod: {retry['total_contours']} kontur bulundu")
            if relaxed.get('total_contours', 0) > 0:
                print(f"      Relaxed mod: {relaxed['total_contours']} kontur bulundu")

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
        # NEW: Load Eye Cascade
        eye_cascade = load_eye_cascade()
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
        # Pass eye_cascade
        process_image(img_path, face_cascade, eye_cascade)
    
    print("\n--- Bitti ---")
    if DEBUG_MODE:
        print(f"[DEBUG] Detaylı görüntüler '{DEBUG_FOLDER}/' klasöründe.")

if __name__ == "__main__":
    main()