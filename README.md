# 👓 LensAI: Real-Time Eyeglass Frame and Lens Detection

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv8-yellow)
![React Native](https://img.shields.io/badge/React_Native-0.73-61DAFB?logo=react)
![Swift](https://img.shields.io/badge/Swift-5.0-F05138?logo=swift)
![Platform](https://img.shields.io/badge/Platform-iOS-lightgrey)

LensAI is a cutting-edge deep learning application designed to detect transparent eyeglass lenses and thin frames with pixel-perfect accuracy. It abandons fragile traditional computer vision methods (like Canny edges and Haar Cascades) in favor of a highly robust **YOLOv8m-seg (Instance Segmentation)** model. 

Crucially, the project features a **Zero-Cloud Architecture**, deploying the heavily trained AI model natively to iOS devices using Apple CoreML and the Neural Engine (ANE) for completely offline, real-time biometric analysis.

---

## ✨ Key Features
- **Data-Driven Instance Segmentation:** Trained on a heavily augmented dataset (892 images) with synthetic glare, blur, and color jittering to handle complex optical physics.
- **Zero-Cloud / Edge Computing:** Completely offline inference. Your biometric data (facial images) never leaves your device.
- **Native Swift Integration:** Bypasses JavaScript bridging latencies in React Native. The iOS Vision Framework and `CVPixelBuffer` interact directly with CoreML for buttery-smooth 40+ FPS performance.
- **Biometric Precision:** Generates continuous vector coordinates (`UIBezierPath`) instead of blocky pixels for high-resolution mask rendering.

---

## 📊 Performance Metrics

The model demonstrates exceptional resilience to optical distortions, easily handling specular reflections that would catastrophically fail traditional edge detectors.

### 1. YOLOv8m-seg Validation Metrics
| Class | Box mAP@50 | Box mAP@50-95 | Mask mAP@50 | Mask mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: |
| **All** | 98.5% | 89.9% | **93.3%** | 68.0% |
| **Glass** | 99.5% | 95.1% | **⭐ 99.5%** | 89.8% |
| **Frame** | 97.5% | 84.7% | **87.2%** | 46.2% |
*(Note: Frame Mask mAP@50-95 is strictly penalized by single-pixel deviations on extremely thin wireframes.)*

### 2. Hardware Inference Latency
| Computing Node | Preprocess | Tensor Inference | Postprocess | Total Latency | Effective FPS |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cloud GPU (NVIDIA L4)** | 0.4 ms | 11.9 ms | 2.4 ms | **14.7 ms** | **~68 FPS** |
| **Edge NPU (iPhone 16)** | - | (Natively Fused) | - | **< 25.0 ms** | **~40 FPS** |
*(Note: Apple Neural Engine (ANE) effectively fuses post-processing (NMS) into the CoreML graph, enabling high-speed offline inference utilizing only ~150MB of RAM.)*

---

## 🚀 How to Use the Models

You can interact with the LensAI models in two ways: either via Python for bulk image analysis on your PC, or via the iOS Mobile App for real-time camera detection.

### Option A: Python Inference (Testing the PyTorch Model)
If you want to run the raw `.pt` model on your computer to test images or videos.

1. **Set up the environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Run Inference via CLI:**
   You can easily test the final model on any image using the Ultralytics CLI:
   ```bash
   yolo predict model="v3_final_results/weights/best.pt" source="path/to/your/test_image.jpg" save=True show=True
   ```
3. **Run Inference via Python Script:**
   ```python
   from ultralytics import YOLO

   # Load the trained model
   model = YOLO("v3_final_results/weights/best.pt")

   # Run prediction
   results = model.predict(source="path/to/image.jpg", conf=0.5, save=True)
   
   # View the exact polygon coordinates
   for r in results:
       if r.masks is not None:
           print("Detected Masks XY Coordinates:", r.masks.xy)
   ```

### Option B: Running the iOS Mobile App (Real-Time CoreML)
The mobile app contains the exported `best.mlpackage` (CoreML) model and runs natively. **A Mac and Xcode are required.**

1. **Navigate to the mobile directory and install Node dependencies:**
   ```bash
   cd mobile
   npm install
   ```
2. **Prebuild the Native iOS Code:**
   This step configures the `Info.plist`, installs iOS CocoaPods, and bridges the Native Swift module to React Native.
   ```bash
   npx expo prebuild -p ios --clean
   ```
3. **Run the Application:**
   ```bash
   npx expo run:ios
   ```
   > **⚠️ CRITICAL NOTE:** The iOS Simulator uses your Mac's CPU to emulate CoreML and will be **very slow**. For the true 40 FPS, <25ms experience, you MUST plug in a physical iPhone via USB and run the app directly on the device to activate the Apple Neural Engine (ANE).

---

## 📂 Project Structure

```text
glass/
├── dataset/             # The fully augmented YOLOv8-seg training dataset (892 images)
├── mobile/              # React Native + Swift iOS Application source code
│   ├── App.tsx          # Main application interface and camera view
│   ├── ios/             # Generated Native iOS Workspace (Xcode)
│   └── modules/         # Custom 'EyeglassDetector' Native Swift module (CoreML Bridge)
├── src/                 # Local model training scripts and dataset utilities
├── v3_final_results/    # The final PyTorch model weights (.pt) and validation graphs
├── COLAB_V4_YOL_HARITASI.md # Documentation/Steps for Cloud Training on Google Colab Pro
├── requirements.txt     # Python dependencies for Ultralytics/PyTorch
└── README.md            # Project documentation (You are here)
```

---

## 🎓 Academic Context
This project serves as a **Graduation Thesis**, demonstrating how modern deep learning techniques (Instance Segmentation) can overcome the severe limitations of heuristic computer vision algorithms (Canny Edge, Haar Cascades) which catastrophically fail under specular reflections (glare) and lens transparency. 

**Core References:**
1. Jocher, G. et al. (2023). "Ultralytics YOLOv8". *Zenodo*.
2. Apple Inc. (2017). "Core ML Framework". *Apple Developer Docs*.
3. He, K. et al. (2017). "Mask R-CNN". *IEEE ICCV*.
