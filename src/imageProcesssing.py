#!/usr/bin/env python3
"""
Aggressive Eyeglass Detection 
----------------------------------------
Priorities:
1. Detection Rate > Aesthetics. (Find it even if it's messy).
2. Heavy Morphological Closing to connect broken frames.
3. Retry Mechanism: If strict parameters fail, retry with relaxed ones.
4. RETR_TREE: Look for inner frame boundaries if outer ones are lost.

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
OUTPUT_FOLDER = 'output_imageprocessing'
DEBUG_FOLDER = 'debug_imageprocessing'
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

def get_precise_pupil_center(roi_gray):
    """
    TIMM-BARTH Gradient-Based Pupil Detection
    "Accurate Eye Centre Localisation by Means of Gradients" (2011)
    
    Finds the point where the most gradient vectors converge (pupil center).
    Robust to illumination changes and low contrast.
    """
    h, w = roi_gray.shape
    if h < 10 or w < 10:
        return w // 2, h // 2
    
    # 1. Downsample for speed (work on smaller image)
    scale = 0.5
    small = cv2.resize(roi_gray, None, fx=scale, fy=scale)
    sh, sw = small.shape
    
    # 2. Calculate gradients (Sobel)
    grad_x = cv2.Sobel(small, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(small, cv2.CV_64F, 0, 1, ksize=3)
    
    # 3. Gradient magnitude for weighting
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Only use strong gradients (threshold)
    grad_thresh = np.percentile(magnitude, 70)
    mask = magnitude > grad_thresh
    
    # Normalize gradients
    magnitude_safe = np.maximum(magnitude, 1e-6)
    gx_norm = grad_x / magnitude_safe
    gy_norm = grad_y / magnitude_safe
    
    # 4. Objective function: Find center where gradients point to
    # Initialize accumulator
    accumulator = np.zeros((sh, sw), dtype=np.float64)
    
    # Create coordinate grids
    Y, X = np.ogrid[:sh, :sw]
    
    # For each candidate center, calculate dot product sum
    for cy in range(2, sh - 2, 2):  # Step by 2 for speed
        for cx in range(2, sw - 2, 2):
            # Vector from (cx, cy) to each pixel
            dx = X - cx
            dy = Y - cy
            dist = np.sqrt(dx**2 + dy**2 + 1e-6)
            
            # Normalize displacement vectors
            dx_norm = dx / dist
            dy_norm = dy / dist
            
            # Dot product: gradient · displacement
            dot = gx_norm * dx_norm + gy_norm * dy_norm
            
            # Only count positive dot products (vectors pointing towards center)
            dot = np.maximum(dot, 0)
            
            # Weight by gradient magnitude and mask
            weighted = dot * magnitude * mask
            
            # Accumulate
            accumulator[cy, cx] = np.sum(weighted)
    
    # 5. Apply Gaussian blur to smooth accumulator
    accumulator = cv2.GaussianBlur(accumulator, (5, 5), 0)
    
    # 6. Find maximum (pupil center)
    _, _, _, max_loc = cv2.minMaxLoc(accumulator)
    
    # Scale back to original size
    pupil_x = int(max_loc[0] / scale)
    pupil_y = int(max_loc[1] / scale)
    
    return pupil_x, pupil_y

def scale_contour(cnt, scale):
    M = cv2.moments(cnt)
    if M['m00'] == 0: return cnt
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    cnt_scaled = cnt_norm * scale
    cnt_scaled = cnt_scaled + [cx, cy]
    return cnt_scaled.astype(np.int32)

def radial_edge_scan(roi, pupil_center, est_radius, guided_mode=False):
    h, w = roi.shape
    px, py = pupil_center
    rim_points = []
    
    # Pre-calculate gradients for edge strength
    grad_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    magnitude = cv2.GaussianBlur(magnitude, (3,3), 0)
    
    # Scan 72 rays (every 5 degrees)
    angles = np.deg2rad(np.arange(0, 360, 5))
    
    if guided_mode:
        # Tighter search, lower threshold
        r_min = int(est_radius * 0.8)
        r_max = int(est_radius * 1.2)
        threshold = 12
    else:
        # Broad search, standard threshold
        r_min = int(est_radius * 0.6)
        r_max = int(est_radius * 1.4)
        threshold = 12
    
    overall_max_val = 0
    
    for theta in angles:
        best_val = 0
        best_pt = None
        
        # Scan along the ray
        for r in range(r_min, r_max, 2):
            x = int(px + r * np.cos(theta))
            y = int(py + r * np.sin(theta))
            
            if 0 <= x < w and 0 <= y < h:
                val = magnitude[y, x]
                if val > best_val:
                    best_val = val
                    best_pt = [x, y]
        
        if best_val > overall_max_val:
            overall_max_val = best_val
            
        # Peak must be strong enough
        if best_pt is not None and best_val > threshold:
            rim_points.append(best_pt)
            
    if len(rim_points) < 5: 
        if DEBUG_MODE:
            print(f"      [FAIL] Radial Scan: Found {len(rim_points)} points (needed 5). Max Grad={overall_max_val:.1f} (Thresh={threshold})")
        return []
    return np.array(rim_points, dtype=np.int32)

def find_frame_candidate(roi_gray, roi_color_crop, eye_cascade=None, sensitivity='normal', debug_prefix='', morph_iters=1, use_mask=False, pupil_center=None, inflate=False, guided_radius=None):
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
    
    # Estimate Frame Scale (Reference)
    if guided_radius is not None:
        est_frame_radius = int(guided_radius)
    else:
        # Glasses are typically ~40-50% of the ROI width in diameter.
        est_frame_radius = int(w_roi * 0.22)

    # 1. Ön İşleme & Kenar Tespiti Stratejisi
    edges = None
    
    if sensitivity == 'metallic':
        # STRATEGY: Adaptive Thresholding on Green Channel
        # Metallic frames often contrast best in Green.
        # Adaptive Threshold finds local differences (thin lines) better than Canny.
        
        # Use Green channel if available, else Gray
        if roi_color_crop is not None:
            # BGR -> G channel is index 1
            source_img = roi_color_crop[:, :, 1]
        else:
            source_img = roi_gray
            
        # Mild blur to keep lines intact but reduce grain
        blurred = cv2.bilateralFilter(source_img, 5, 50, 50)
        
        # High contrast CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(blurred)
        
        # Adaptive Thresholding: Finds dark lines on light background
        # Block size 15, C=3 (Tuned for thin frames)
        edges = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, 15, 3)
        
        # Filter out tiny noise specks (salt-and-pepper noise)
        edges = cv2.medianBlur(edges, 3)
        
        if DEBUG_MODE and debug_prefix:
            cv2.imwrite(str(Path(DEBUG_FOLDER) / f"{debug_prefix}_edges_adaptive_raw.png"), edges)
            
    else:
        # STRATEGY: Standard Canny (Normal/Relaxed)
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
            # YENİ: get_precise_pupil_center ile sağlam göz bebeği tespiti
            # (Glare removal + Morph opening + Gaussian center bias)
            rel_x, rel_y = get_precise_pupil_center(center_roi)
            pupil_x = rel_x + margin_w
            pupil_y = rel_y + margin_h
                 
        if DEBUG_MODE and debug_prefix:
            # DEBUG: Göz tespitinin sonucunu hemen göster
            debug_path = Path("debug")
            debug_path.mkdir(exist_ok=True)
            pupil_vis = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            cv2.circle(pupil_vis, (pupil_x, pupil_y), 5, (0, 0, 255), -1)
            cv2.imwrite(str(debug_path / f"{debug_prefix}_01_pupil_loc.png"), pupil_vis)

        if use_mask:
            # DYNAMIC PUPIL BLURRING (Akıllı Göz Bulanıklaştırma)
            # Bulduğumuz pupil noktasına göre devasa alanı blurla.
            
            # 1. Yumuşak Maske Oluştur
            # CHANGE: Alan son kez ayarlandı (%18 Width, %16 Height)
            # User Feedback: "genişlik 18 yükseklik 16 olsun"
            axes = (int(w_roi * 0.18), int(h_roi * 0.15))
            
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
    
    # Use pupil_center if provided (for metallic mode)
    if pupil_center is not None:
        pupil_x, pupil_y = pupil_center
    elif sensitivity != 'metallic':
        # For non-metallic modes, pupil was calculated above
        pass
    else:
        # Metallic mode fallback
        pupil_x, pupil_y = w_roi // 2, h_roi // 2
    
    # 3. Morfolojik Kapatma (Boşluk Doldurma)
    # Çerçeve kırıksa birleştir. Yatay kernel kullanıyoruz.
    if sensitivity == 'metallic':
        k_size = (3, 3)  # For adaptive threshold, lines are already "thick"
    else:
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
    
    # --- GEOMETRIC COMPLETION (For Metallic Mode) ---
    # User: "Predict glass frame using estimated diameter as reference"
    if sensitivity == 'metallic' and pupil_center is not None and candidates:
        px, py = pupil_center
        rim_points = []
        
        # Look at more candidates than just the top one
        for _, cnt in candidates[:10]:
            # Calculate distance of this fragment from pupil
            mx, my = cv2.boundingRect(cnt)[0:2]
            dist = np.sqrt((mx - px)**2 + (my - py)**2)
            
            # If fragment is roughly on the predicted rim (60% to 200% of expected radius)
            # CHANGE: Increased min distance from 0.4 to 0.6 to avoid detecting the eye itself
            if est_frame_radius * 0.6 < dist < est_frame_radius * 2.0:
                rim_points.append(cnt)
        
        # Merge points from valid contours
        combined_points = []
        if rim_points:
            combined_points = np.vstack(rim_points)
            
        # FALLBACK: If contours didn't give enough points, try Radial Scan (Direct Edge Search)
        if len(combined_points) < 50: # Arbitrary threshold for "enough evidence"
            # Use raw gray image for scanning
            # Enable guided mode if we have a specific target radius
            scan_points = radial_edge_scan(roi_gray, (px, py), est_frame_radius, guided_mode=(guided_radius is not None))
            if len(scan_points) > 0:
                # Convert scan_points to contour format (3D: [n, 1, 2])
                scan_points_3d = scan_points.reshape(-1, 1, 2)
                if len(combined_points) > 0:
                    combined_points = np.vstack([combined_points, scan_points_3d])
                else:
                    combined_points = scan_points_3d
                    
        if len(combined_points) >= 5:
            # CHANGE: Use Convex Hull instead of FitEllipse
            # User: "Right frame is plain circle, no corners... looks hand drawn"
            # Convex Hull connects the actual points, preserving corners/shape.
            hull_geo = cv2.convexHull(combined_points)
            
            # Simple sanity check on size
            x,y,w,h = cv2.boundingRect(hull_geo)
            if w > est_frame_radius * 0.5 and h > est_frame_radius * 0.5:
                best_cnt = hull_geo
                if DEBUG_MODE and debug_prefix:
                    print(f"      [GEO] {debug_prefix}: Completed frame using ConvexHull of {len(combined_points)} points.")
            else:
                best_cnt = candidates[0][1]
        else:
            best_cnt = candidates[0][1]
    elif candidates:
        # En iyi adayı al (Normal/Relaxed)
        best_cnt = candidates[0][1]
    else:
        return None, debug_info
    
    # User: "Kopsa bile birleştir" -> Convex Hull uygula
    hull = cv2.convexHull(best_cnt)
    
    # --- INFLATION STRATEGY (User: "Left frame is small/around eye") ---
    # We found the inner rim. Expand slightly to cover the frame thickness.
    if inflate:
        # Conservative expansion (15%) to match outer rim
        inflation_factor = 1.15
        
        # Dynamic: If it's surprisingly small, inflate a bit more
        hull_area = cv2.contourArea(hull)
        if hull_area < roi_area * 0.15:
            inflation_factor = 1.25 
            
        hull = scale_contour(hull, inflation_factor)
    
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

def process_single_eye_roi(roi_gray, roi_color_crop, eye_cascade, full_image, offset_x, offset_y, debug_prefix='', forced_iters=None, guided_radius=None):
    if roi_gray.size == 0: 
        return 0, {}, 0

    h_roi, w_roi = roi_gray.shape
    roi_area = h_roi * w_roi
    
    # Pre-calculate pupil center for all modes (Reference for Metallic)
    # Temporary CLAHE for pupil finding
    tmp_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    tmp_enhanced = tmp_clahe.apply(cv2.bilateralFilter(roi_gray, 9, 75, 75))
    
    margin_h = int(h_roi * 0.30)
    margin_w = int(w_roi * 0.30)
    center_roi = tmp_enhanced[margin_h:-margin_h, margin_w:-margin_w]
    
    pupil_x, pupil_y = w_roi // 2, h_roi // 2
    if center_roi.size > 0:
        rel_x, rel_y = get_precise_pupil_center(center_roi)
        pupil_x = rel_x + margin_w
        pupil_y = rel_y + margin_h
    
    pupil_center = (pupil_x, pupil_y)

    # Determine starting iterations
    # If forced_iters is provided (by Symmetry Check), use it.
    # Otherwise default to 1 (Safe mode).
    start_iters = forced_iters if forced_iters is not None else 1

    # PLAN A: Normal Hassasiyet
    # Eğer forced_iters varsa (Symmetry Retry), doğrudan Relaxed modda çalış.
    # Çünkü iter=2 kalın çerçeve yapar, normal modun filtrelerine takılır.
    plan_a_sensitivity = 'relaxed' if forced_iters else 'normal'
    
    # Default: inflate=False (Keep normal images normal)
    best_cnt, debug_info_normal = find_frame_candidate(
        roi_gray, roi_color_crop=roi_color_crop, eye_cascade=eye_cascade, sensitivity=plan_a_sensitivity, debug_prefix=f"{debug_prefix}_normal", 
        morph_iters=start_iters, pupil_center=pupil_center, inflate=False
    )
    
    # Validation: If Plan A result is weak (too small), discard it to try harder modes
    if best_cnt is not None:
        if cv2.contourArea(best_cnt) < roi_area * 0.10:
            best_cnt = None
            debug_info_normal['discarded'] = 'too_small_force_next'
    
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
    if best_cnt is None and forced_iters is None:
        # Only run relaxed if we aren't ALREADY running forced relaxed
        best_cnt, debug_info_relaxed = find_frame_candidate(
            roi_gray, roi_color_crop=roi_color_crop, eye_cascade=eye_cascade, sensitivity='relaxed', debug_prefix=f"{debug_prefix}_relaxed", 
            morph_iters=start_iters + 1, pupil_center=pupil_center, inflate=False
        )
        if best_cnt is not None:
            if cv2.contourArea(best_cnt) < roi_area * 0.10:
                best_cnt = None
                debug_info_relaxed['discarded'] = 'too_small_force_metallic'

    # PLAN C: Metallic Mode (The "Prediction" Mode)
    debug_info_metallic = {}
    if best_cnt is None:
        # Run this even if forced_iters is set (Symmetry Retry should use this new power!)
        # INFLATE IS ENABLED HERE (Only for the hard cases)
        # Pass guided_radius if available
        best_cnt, debug_info_metallic = find_frame_candidate(
            roi_gray, roi_color_crop=roi_color_crop, eye_cascade=eye_cascade, sensitivity='metallic', debug_prefix=f"{debug_prefix}_metallic", 
            morph_iters=start_iters + 1, pupil_center=pupil_center, inflate=True, guided_radius=guided_radius
        )
        
    found_area = 0
    best_cnt_offset = None
    if best_cnt is not None:
        found_area = cv2.contourArea(best_cnt)
        best_cnt_offset = best_cnt + [offset_x, offset_y]
        
        mode_used = 'normal'
        if debug_info_metallic.get('total_contours', 0) > 0: mode_used = 'metallic'
        elif debug_info_relaxed.get('total_contours', 0) > 0: mode_used = 'relaxed'
        
        combined_debug = {
            'mode_used': mode_used,
            'normal': debug_info_normal,
            'retry': debug_info_retry,
            'relaxed': debug_info_relaxed,
            'metallic': debug_info_metallic
        }
        return 1, combined_debug, found_area, best_cnt_offset, pupil_center
            
    combined_debug = {
        'mode_used': 'failed',
        'normal': debug_info_normal,
        'retry': debug_info_retry,
        'relaxed': debug_info_relaxed,
        'metallic': debug_info_metallic
    }
    return 0, combined_debug, 0, None, pupil_center

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
        
        # Static Strip ROI (sabit bölme - en stabil yöntem)
        coords = [
            (fx, roi_y_start, roi_x_mid, roi_y_end, fx, roi_y_start, 'left'),
            (roi_x_mid, roi_y_start, fx+fw, roi_y_end, roi_x_mid, roi_y_start, 'right')
        ]

        for (x1, y1, x2, y2, off_x, off_y, side) in coords:
            y1=max(0, y1); y2=min(gray.shape[0], y2)
            x1=max(0, x1); x2=min(gray.shape[1], x2)
            
            roi_g = gray[y1:y2, x1:x2]
            roi_c = img[y1:y2, x1:x2]
            
            if roi_g.size == 0 or roi_g.shape[0] < 10 or roi_g.shape[1] < 10:
                continue
            
            debug_prefix = f"{input_path.stem}_face{face_idx}_{side}" if DEBUG_MODE else ""
            
            # 1. INITIAL PASS
            hulls_found, debug_info, area, cnt_offset, pupil = process_single_eye_roi(roi_g, roi_c, eye_cascade, img, off_x, off_y, debug_prefix)
            
            # Store everything needed for potential retry
            eye_args = (roi_g, roi_c, eye_cascade, img, off_x, off_y, debug_prefix)
            eye_results[face_idx][side] = {
                'found': hulls_found,
                'debug': debug_info,
                'area': area,
                'cnt': cnt_offset,
                'pupil': pupil,
                'args': eye_args
            }
            
    # 2. SYMMETRY CHECK & RETRY
    for face_idx, sides in eye_results.items():
        if 'left' in sides and 'right' in sides:
            l_res = sides['left']
            r_res = sides['right']
            
            l_cnt = l_res.get('cnt')
            r_cnt = r_res.get('cnt')
            
            def get_radius(cnt):
                if cnt is None: return 0
                _, r = cv2.minEnclosingCircle(cnt)
                return r

            l_rad = get_radius(l_cnt)
            r_rad = get_radius(r_cnt)
            
            target_side = None
            guided_r = 0
            
            # Logic: If one is good and other is missing or < 60% of good one
            if l_cnt is not None and (r_cnt is None or r_rad < l_rad * 0.6):
                target_side = 'right'
                guided_r = l_rad
            elif r_cnt is not None and (l_cnt is None or l_rad < r_rad * 0.6):
                target_side = 'left'
                guided_r = r_rad
                
            if target_side:
                if DEBUG_MODE:
                    print(f"      [SYMMETRY] {filename} Face {face_idx}: {target_side.upper()} missing/small -> Guided Retry with Radius={guided_r:.1f}...")
                
                target_res = sides[target_side]
                args = target_res['args']
                
                new_found, new_debug, new_area, new_cnt, new_pupil = process_single_eye_roi(
                    *args, forced_iters=2, guided_radius=guided_r
                )
                
                if new_found:
                    sides[target_side]['found'] = new_found
                    sides[target_side]['debug'] = new_debug
                    sides[target_side]['area'] = new_area
                    sides[target_side]['cnt'] = new_cnt
                    sides[target_side]['pupil'] = new_pupil
                    sides[target_side]['debug']['symmetry_retry'] = True

    # 3. CONSOLIDATE RESULTS & DRAW
    for face_idx, sides in eye_results.items():
        for side, res in sides.items():
            total_hulls += res['found']
            debug_info = res['debug']
            
            # DRAW FINAL CONTOUR ONLY ONCE HERE
            if res['cnt'] is not None:
                cv2.drawContours(img, [res['cnt']], -1, (255, 255, 0), 2)

            if DEBUG_MODE:
                debug_info['side'] = side
                debug_info['face_idx'] = face_idx
                all_debug_info.append(debug_info)

    output_path = Path(OUTPUT_FOLDER) / f"{input_path.stem}_aggressive{input_path.suffix}"
    cv2.imwrite(str(output_path), img)
    
    if total_hulls > 0:
        print(f"[BASARILI] {filename} -> {total_hulls} cam bulundu.")
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
            metallic = dbg.get('metallic', {})
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
            if metallic.get('total_contours', 0) > 0:
                print(f"      Metallic mod: {metallic['total_contours']} kontur bulundu")

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