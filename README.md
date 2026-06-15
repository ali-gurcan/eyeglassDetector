# LensAI: Real-Time Eyeglass Frame and Lens Detection

This project is a comprehensive deep learning application aimed at detecting eyeglass frames and lenses on human faces and deploying this capability in real-time on a mobile environment (iOS).

Legacy traditional computer vision algorithms (such as Haar Cascades, Canny edge detection) have been entirely abandoned. The project is now solely built upon a **Custom Trained YOLOv8m-seg (Instance Segmentation)** model and a native iOS application.

## 🎯 Project Features

* **Custom Trained Segmentation Model:** Using a heavily augmented dataset of 892 eyeglass images, the YOLOv8m-seg (v4) model was trained from scratch. It is capable of detecting lens (`glass`) and frame (`frame`) boundaries with pixel-level precision.
* **Apple CoreML Integration:** To fully utilize the machine learning hardware (Neural Engine/NPU) on iOS devices, the PyTorch-trained YOLOv8m-seg network was converted to the `best.mlpackage` (CoreML) format.
* **React Native & Swift iOS App:** The detection process runs entirely on-device without requiring an internet connection. The frontend UI is built with React Native (Expo), while CoreML inference and rendering are handled by a highly optimized **Native Swift** module to maximize performance and FPS.
* **Gallery and Camera Integration:** The application allows users to capture live photos or pick existing photos from their gallery for analysis. Analyzed images with drawn boundaries can be saved directly back to the device's photo gallery.

## 📊 Dataset and Training (v3 and v4 Models)

* **Dataset Size:** By applying synthetic data augmentation techniques to base images, a heavily enriched dataset (V4 Dataset) consisting of **892** images was generated. The classes are `frame` and `glass`.
* **Synthetic Data Augmentation:** To ensure the model is robust against real-world challenging conditions, manipulations such as artificial glare/reflections (`glare`), color jittering (`color`), and blurring (`blur`) were applied to generate the augmented dataset.
* **v3 Training (Local Prototype):** Initially, a YOLOv8n-seg model was trained for 150 epochs locally on Apple Silicon (MPS hardware acceleration) using `src/train.py` to establish a working prototype.
* **v4 Training (Colab Pro - L4 GPU):** The final, production-ready model was trained on **Google Colab Pro** utilizing a powerful **NVIDIA L4 GPU (22GB VRAM)**. This intensive training used the **YOLOv8m-seg** (Medium) architecture, ran for **200 epochs** with an AutoBatch of 4, and utilized a high resolution (`imgsz=1024`) for extreme precision. The model achieved an outstanding mAP@50 of ~99% during validation.
* **Results:** All training metrics, loss curves, and the exported CoreML packages (`best.mlpackage`) are stored for iOS integration.

## 🛠️ Tech Stack

* **Machine Learning:** Python, PyTorch, Ultralytics YOLOv8.
* **Mobile Deployment:** Apple CoreML, Vision Framework (iOS Native).
* **Mobile App (Frontend):** React Native, Expo, TypeScript.
* **Mobile Module (Native):** Swift, Objective-C.

## 📂 Project Structure

```text
glass/
│
├── auto_retina_dataset/ # 892-image training dataset heavily augmented with glare and blur
├── manipulated_images/  # Raw source folder for images with glare, blur, and color filters
├── dataset/             # The fully augmented YOLOv8-seg training dataset (892 images)
├── src/                 # Local model training script (train.py) and dataset tools
├── v3/                  # The initial YOLOv8-seg prototype model
├── COLAB_V4_YOL_HARITASI.md # Documentation/Steps for the Google Colab Pro (L4) V4 training
├── runs/                # YOLOv8 training metrics, loss graphs, and evaluation results
├── mobile/              # React Native + Swift iOS Application source code
│   ├── App.tsx          # Main application interface and navigation
│   └── modules/         # Custom 'EyeglassDetector' Native Swift module
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

## 🚀 Installation and Usage

### 1. Running the iOS Application (Requires Mac and Xcode)

The mobile application consists of an Expo frontend and Native Swift modules.

```bash
cd mobile
# Install Node dependencies
npm install

# Prebuild native iOS packages (Configures Info.plist and Podfile)
npx expo prebuild -p ios --clean

# Run the app on the iOS Simulator or a connected physical iPhone
npx expo run:ios
```

> **Note:** For maximum performance, test the application on a physical iPhone. The Neural Engine (NPU) handles CoreML inference much faster than the CPU-based iOS Simulator.

### 2. Retraining the Model (Optional)

You can set up a Python virtual environment and run the local `train.py` script.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start YOLOv8 local training
python src/train.py
```

## 📱 Mobile Application & Thesis Screenshots

The iOS mobile application acts as the real-time inference engine. Built with **React Native** and a custom **Native Swift Module**, the app captures camera frames, pre-processes the CVPixelBuffer, runs it through the Apple Neural Engine using CoreML, and renders `UIBezierPath` mask polygons directly onto the screen.

### 📸 Required Screenshots for the Thesis
To complete your LaTeX Graduation Thesis, you need to capture the following screenshots from the iOS app and place them in the `Graduation Project - Latex Template/Imgs/` directory:

1. **Successful Detection 1 (`app_success_1.png`):** A screenshot of the app successfully masking a standard pair of glasses in a well-lit environment. *(Goes to Chapter 4)*
2. **Successful Detection 2 (`app_success_2.png`):** A screenshot of the app correctly identifying glasses under challenging conditions (e.g., glare or low light). *(Goes to Chapter 4)*
3. **Error Case (`error_case.png`):** A screenshot where the model slightly struggles, such as frameless/rimless glasses blending into the background, or highly reflective mirrored sunglasses. *(Goes to Chapter 4)*

*(Note: You will also need your Colab training loss graphs `training_results.png` for Chapter 4).*

## 🎓 Academic Scope

This project serves as a **Graduation Thesis**, demonstrating how modern deep learning techniques (specifically Instance Segmentation) can overcome the limitations of traditional computer vision methods (Haar Cascades, Canny Edge, etc.). It showcases the end-to-end pipeline of training a robust AI model and deploying it on a mobile edge device utilizing hardware-accelerated CoreML for real-time inference.
