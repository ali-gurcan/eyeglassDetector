# AI-Powered Eyeglass Lens Edge Detection

---

## 1. Project Definition

A smartphone app for opticians that detects the precise lens edges of eyeglasses from a single photo. It combines deep learning models with digital image processing techniques (edge detection, morphological operations, contour analysis) to accurately find where each lens meets the frame. The optician photographs the customer and instantly sees the result — fully offline, no internet needed.

```plantuml
@startuml
skinparam backgroundColor transparent
skinparam shadowing false
skinparam defaultFontName "Trebuchet MS"
skinparam defaultFontSize 14
skinparam defaultFontColor #2D3436

skinparam rectangle {
  RoundCorner 20
  BorderThickness 1.5
  FontSize 15
  FontStyle bold
}

skinparam arrow {
  Color #636E72
  Thickness 2
  FontSize 12
  FontColor #555555
}

rectangle "Customer\nWears Glasses" as input #E8DAEF/D2B4DE
rectangle "Optician Takes\na Photo" as cam #D6EAF8/AED6F1
rectangle "App Detects\nLens Edges" as process #D5F5E3/ABEBC6
rectangle "Optician Sees\nthe Result" as output #FDEBD0/F5CBA7

input -right-> cam : Photographed
cam -down-> process : Analyzes
process -left-> output : Shows edges
@enduml
```

![Project Concept](assets/image-f4901ea9-786d-4797-85bc-0814e0351aa7.png)
*(Figure 1: Conceptual overview showing user input, on-device processing, and visual output)*

---

## 2. Project Design

The system runs entirely on the smartphone with no server or internet dependency. It follows a three-stage pipeline: first, a lightweight object detector (YOLOv8-Nano) locates the glasses in the photo; then, a mobile segmentation model (MobileSAM) isolates the frame and lens regions; finally, classical image processing algorithms (Canny edge detection, contour analysis) extract the precise lens boundary. The diagram below illustrates these five components and the data flowing between them.

### System Architecture

```plantuml
@startuml
skinparam backgroundColor transparent
skinparam shadowing false
skinparam defaultFontName "Trebuchet MS"
skinparam defaultFontColor #2D3436

skinparam rectangle {
  RoundCorner 15
  BorderThickness 1.5
  BorderColor #95A5A6
  FontSize 14
}

skinparam arrow {
  Color #636E72
  Thickness 2
  FontSize 12
  FontColor #555555
}

actor "Optician" as user

package "Smartphone (Offline)" #F0F0F0 {
  rectangle "Camera" as cam #D6EAF8
  rectangle "Glasses Detector\n(YOLOv8-Nano)" as det #E8DAEF
  rectangle "Lens Segmenter\n(MobileSAM)" as sam #D5F5E3
  rectangle "Edge Refiner\n(OpenCV)" as edge #FDEBD0
  rectangle "Result Display" as ui #FADBD8
}

user -down-> cam : Customer photo
cam -right-> det : Image
det -down-> sam : Bounding box
sam -left-> edge : Mask
edge -down-> ui : Lens contour
ui -up-> user : Result
@enduml
```

### Data Flow
1. **Input:** Optician captures a photo of the customer wearing glasses via the app camera.
2. **Detection:** A small object detection model locates the glasses on the customer's face.
3. **Segmentation:** MobileSAM generates a precise mask of the frame using the detected location.
4. **Refinement:** Classical computer vision algorithms extract the exact inner lens edge from the mask.
5. **Output:** The final lens contour is displayed to the optician in real-time.

---

## 3. Project Requirements

### Software
- **Mobile Framework:** React Native / Expo (for cross-platform app development)
- **AI Inference:** ONNX Runtime Mobile / TensorFlow Lite (to run models on-device)
- **Computer Vision:** OpenCV Mobile (for image processing algorithms)
- **Models:** MobileSAM (Segment Anything Model), YOLOv8-Nano (for detection)

### Hardware
- **Development:** macOS/Linux workstation with GPU for model training/optimization.
- **Target Device:** Modern smartphone (iOS 15+ or Android 10+) with camera and NPU (Neural Processing Unit) support for fast inference.

### Other Resources
- **Dataset:** Proprietary dataset of 40+ high-resolution eyeglass images for testing.
- **Public Datasets:** Roboflow & Kaggle eyeglass datasets for model training.

---

## 4. Success Criteria

1. **Lens Detection ≥ 85%:** Given a photo of a customer wearing eyeglasses, the system must correctly detect and outline both lenses in at least 85 out of 100 images. *Test: Run the on-device pipeline on the test set and count successful detections.*

2. **On-Device Latency ≤ 3 seconds:** The total processing time (Detection + SAM + Refinement) must be under 3 seconds on a mid-range smartphone. *Test: Measure end-to-end inference time on a physical device (e.g., iPhone 13).*

3. **Works Offline:** The app must function 100% without an internet connection, usable in any optical shop. *Test: Enable airplane mode and verify that the entire pipeline (photo to result) works without errors.*

---

## 5. Project Timeline

| Month | Task | Deliverable |
| :--- | :--- | :--- |
| **Month 1** | Research & Model Selection (MobileSAM vs SAM 2 Tiny) | Performance Benchmark Report |
| **Month 2** | Desktop Pipeline Development (Simulating Mobile Constraints) | Working Python Prototype |
| **Month 3** | Convert Models for Mobile (ONNX export, quantization, size & speed testing) | Optimized lightweight models ready for smartphone |
| **Month 4** | Mobile App Development (React Native + ONNX Runtime) | Alpha App with Camera |
| **Month 5** | Integration (Connecting Detector -> SAM -> OpenCV on Mobile) | Beta App (Full Pipeline) |
| **Month 6** | Performance Tuning & UI Polish | Release Candidate (RC) |

---

## 6. References

1. Ravi, N. et al. — *SAM 2: Segment Anything in Images and Videos.* Meta FAIR, 2024. [github.com/facebookresearch/sam2](https://github.com/facebookresearch/sam2)
2. Zhang, C. et al. — *MobileSAM: Lightweight Segment Anything Model.* 2023. [github.com/ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
3. ONNX Runtime — *Cross-platform, high performance ML inference.* [onnxruntime.ai](https://onnxruntime.ai)
4. Canny, J. — *A Computational Approach to Edge Detection.* IEEE TPAMI, 1986.
5. React Native — *Cross-platform mobile framework.* [reactnative.dev](https://reactnative.dev)
