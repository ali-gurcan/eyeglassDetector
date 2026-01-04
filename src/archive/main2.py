#!/usr/bin/env python3
"""
Symmetric Eyeglass Detection (Mirroring & Inner Guessing)
----------------------------------------
Logic:
1. Detect left and right candidates independently.
2. Score them based on geometry (Area, Aspect Ratio, Solidity).
3. If one side is GOOD and the other is BAD/MISSING:
   -> Mirror the good side relative to the face center.
4. "Guess" the lens area by shrinking (eroding) the frame contour.

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

def get_best_contour_and_score(roi_gray):
    """
    Returns the best contour and its quality score.
    Score 0 = Trash, Score 100 = Perfect Frame.
    """
    if roi_gray.size == 0: return None, 0
    
    # 1. Pre-processing
    blurred = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    
    # 2. Edge Detection (Auto Canny)
    v = np.median(enhanced)
    lower = int(max(0, (1.0 - 0.33) * v))
    upper = int(min(255, (1.0 + 0.33) * v))
    edges = cv2.Canny(enhanced, lower, upper)
    
    # 3. Closing (Connect gaps)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    h_roi, w_roi = roi_gray.shape
    roi_area = h_roi * w_roi
    roi_center_y = h_roi // 2
    
    best_cnt = None
    best_score = 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = float(w) / h
        
        # --- Quality Checks ---
        if area < roi_area * 0.05: continue
        if area > roi_area * 0.60: continue
        if not (1.0 < aspect < 4.0): continue # Glasses are horizontal
        
        # Anti-Eyebrow (Too high up)
        if y < h_roi * 0.10: continue
        
        # --- Scoring ---
        # 1. Solidity (Düzenlilik): Çerçeveler düzgündür
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        
        # 2. Centrality (Merkezilik)
        cy = y + h // 2
        dist_y = abs(cy - roi_center_y)
        centrality = 1.0 - (dist_y / h_roi)
        
        # Score Formula
        score = (solidity * 50) + (centrality * 50)
        
        if score > best_score:
            best_score = score
            best_cnt = hull # Use hull for smooth shape
            
    return best_cnt, best_score

def mirror_contour(cnt, face_width):
    """
    Mirrors a contour from left to right (or vice versa) relative to face center.
    """
    if cnt is None: return None
    
    # Y ekseni etrafında çevir (Flip)
    # Yeni X = (FaceWidth) - Eski X
    mirrored = []
    for point in cnt:
        x, y = point[0]
        new_x = face_width - x
        mirrored.append([[new_x, y]])
        
    return np.array(mirrored, dtype=np.int32)

def process_image(img_path, face_cascade):
    input_path = Path(img_path)
    filename = input_path.name
    img = cv2.imread(str(input_path))
    if img is None: return

    # Standardize size
    if img.shape[1] > 1000:
        scale = 1000 / img.shape[1]
        img = cv2.resize(img, None, fx=scale, fy=scale)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    # Fallback face
    if len(faces) == 0:
        h, w = img.shape[:2]
        faces = [[int(w*0.25), int(h*0.2), int(w*0.5), int(h*0.6)]]
    else:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    for (fx, fy, fw, fh) in faces:
        # Define Eye Region (Strip)
        roi_y_start = int(fy + fh * 0.18)
        roi_y_end = int(fy + fh * 0.55)
        face_roi_h = roi_y_end - roi_y_start
        
        # Cut the strip from gray image
        face_strip_gray = gray[roi_y_start:roi_y_end, fx:fx+fw]
        
        # Split Strip into Left and Right
        mid_x = fw // 2
        left_roi = face_strip_gray[:, 0:mid_x]
        right_roi = face_strip_gray[:, mid_x:fw]
        
        # 1. DETECT BOTH SIDES
        cnt_left, score_left = get_best_contour_and_score(left_roi)
        cnt_right, score_right = get_best_contour_and_score(right_roi)
        
        # 2. APPLY SYMMETRY LOGIC (Mirroring)
        final_left = cnt_left
        final_right = cnt_right
        
        # Threshold score to decide if detection is "Good"
        GOOD_SCORE = 60 
        
        # Scenario A: Left is good, Right is bad -> Mirror Left to Right
        if score_left > GOOD_SCORE and score_right < GOOD_SCORE:
            print(f"[{filename}] Mirroring Left -> Right")
            # Mirror logic: Flip X within the half-width
            mirrored_shape = mirror_contour(cnt_left, fw) # Mirror across full face width
            # Adjust to right ROI coordinates
            # Since mirror_contour flips across full width, the result is already in "Right" space relative to face origin
            # We need to map it back to global image later.
            # Actually simpler: Mirror relative to face center.
            final_right = []
            for pt in cnt_left:
                lx, ly = pt[0]
                rx = fw - lx # Mirror across center
                final_right.append([[rx, ly]])
            final_right = np.array(final_right, dtype=np.int32)

        # Scenario B: Right is good, Left is bad -> Mirror Right to Left
        elif score_right > GOOD_SCORE and score_left < GOOD_SCORE:
            print(f"[{filename}] Mirroring Right -> Left")
            final_left = []
            for pt in cnt_right:
                rx, ry = pt[0] # Note: contours are relative to their ROI, need adjustment
                # Right ROI starts at mid_x.
                # Let's verify coordinates first.
                # Easier way: Convert everything to "Face Strip" coordinates first.
                pass 

        # --- Coordinate Standardization for Drawing ---
        draw_list = []
        
        # Process Left
        if final_left is not None:
            # Adjust to Global Image Coordinates
            # Left ROI x starts at fx
            global_cnt = final_left + [fx, roi_y_start]
            draw_list.append(global_cnt)
            
            # If we need to mirror THIS to the right:
            if score_left > GOOD_SCORE and score_right < GOOD_SCORE:
                # Mirror global points around face center line
                face_center_x = fx + fw // 2
                mirrored_cnt = []
                for pt in global_cnt:
                    gx, gy = pt[0]
                    dist = face_center_x - gx
                    new_x = face_center_x + dist
                    mirrored_cnt.append([[new_x, gy]])
                draw_list.append(np.array(mirrored_cnt, dtype=np.int32))

        # Process Right
        if final_right is not None and not (score_left > GOOD_SCORE and score_right < GOOD_SCORE):
            # Adjust to Global Image Coordinates
            # Right ROI x starts at fx + mid_x
            global_cnt = final_right + [fx + mid_x, roi_y_start]
            draw_list.append(global_cnt)
            
            # If we need to mirror THIS to the left:
            if score_right > GOOD_SCORE and score_left < GOOD_SCORE:
                face_center_x = fx + fw // 2
                mirrored_cnt = []
                for pt in global_cnt:
                    gx, gy = pt[0]
                    dist = gx - face_center_x
                    new_x = face_center_x - dist
                    mirrored_cnt.append([[new_x, gy]])
                draw_list.append(np.array(mirrored_cnt, dtype=np.int32))

        # 3. DRAWING & "INNER GUESSING" (Erosion)
        for cnt in draw_list:
            # Draw Frame (Outer)
            cv2.drawContours(img, [cnt], -1, (255, 255, 0), 2, cv2.LINE_AA)
            
            # --- INNER GUESS ---
            # "Erode" the contour to shrink it and find the lens
            # We create a mask, draw the filled contour, erode the mask, find new contour
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            
            # Erosion Amount: Depends on frame thickness guess (e.g., 5-8 pixels)
            kernel_erode = np.ones((7, 7), np.uint8)
            eroded_mask = cv2.erode(mask, kernel_erode, iterations=1)
            
            inner_contours, _ = cv2.findContours(eroded_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for ic in inner_contours:
                # Draw Lens (Inner) - Green
                cv2.drawContours(img, [ic], -1, (0, 255, 0), 2, cv2.LINE_AA)

    output_path = Path(OUTPUT_FOLDER) / f"{input_path.stem}_symmetric{input_path.suffix}"
    cv2.imwrite(str(output_path), img)
    print(f"Saved: {output_path.name}")

def main():
    print("--- Starting SYMMETRIC PIPELINE ---")
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