from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(BASE_DIR, 'images')
OUTPUT_DIR = os.path.join(BASE_DIR, 'runs/segment/glass_project/debug_final_results')
MODEL_PATH = os.path.join(BASE_DIR, 'runs/segment/glass_project/v2_high_res_small2/weights/best.pt')

os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    model = YOLO(MODEL_PATH)
    print(f"Model yüklendi: {MODEL_PATH}")
except Exception as e:
    print(f"Model yüklenemedi: {e}")
    model = None

COLORS = {
    "frame": (128, 0, 0),     # navy blue
    "glass": (0, 255, 255),   # cyan
}

@app.route('/gallery', methods=['GET'])
def get_gallery():
    image_exts = ("*.png", "*.jpg", "*.jpeg")
    image_paths = []
    for ext in image_exts:
        image_paths.extend(glob.glob(os.path.join(IMAGES_DIR, ext)))
    
    host = request.host_url
    results = []
    for p in sorted(image_paths):
        fname = os.path.basename(p)
        results.append({
            "id": fname,
            "uri": f"{host}image/raw/{fname}"
        })
    return jsonify(results)

@app.route('/gallery/processed', methods=['GET'])
def get_processed_gallery():
    image_exts = ("*.png", "*.jpg", "*.jpeg")
    image_paths = []
    for ext in image_exts:
        image_paths.extend(glob.glob(os.path.join(OUTPUT_DIR, ext)))
    
    host = request.host_url
    results = []
    for p in sorted(image_paths):
        fname = os.path.basename(p)
        # Original id could be inferred but let's just return the processed uri
        results.append({
            "id": fname,
            "uri": f"{host}image/processed/{fname}"
        })
    return jsonify(results)

@app.route('/image/raw/<path:filename>', methods=['GET'])
def get_raw_image(filename):
    return send_from_directory(IMAGES_DIR, filename)

@app.route('/image/processed/<path:filename>', methods=['GET'])
def get_processed_image(filename):
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/analyze/<filename>', methods=['POST'])
def analyze_image(filename):
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
        
    img_path = os.path.join(IMAGES_DIR, filename)
    if not os.path.exists(img_path):
        return jsonify({"error": "Image not found"}), 404
        
    img = cv2.imread(img_path)
    if img is None:
        return jsonify({"error": "Cannot read image"}), 500

    results = model.predict(img, conf=0.25, verbose=False, retina_masks=True, imgsz=1024)
    result = results[0]

    out = img.copy()
    overlay = img.copy()

    if result.masks is not None:
        for i, mask in enumerate(result.masks.xy):
            cls_id = int(result.boxes.cls[i])
            cls_name = result.names[cls_id]
            conf = float(result.boxes.conf[i])
            color = COLORS.get(cls_name, (255, 255, 255))
            pts = np.array(mask, dtype=np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(overlay, [pts], color)
                cv2.drawContours(out, [pts], -1, color, 2)
                
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                label = f"{cls_name} {conf:.0%}"
                cv2.putText(out, label, (cx - 30, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                
    alpha = 0.3
    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)
    
    stem, ext = os.path.splitext(filename)
    out_filename = f"{stem}_trained{ext}"
    out_path = os.path.join(OUTPUT_DIR, out_filename)
    cv2.imwrite(out_path, out)
    
    host = request.host_url
    return jsonify({
        "resultUri": f"{host}image/processed/{out_filename}"
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
