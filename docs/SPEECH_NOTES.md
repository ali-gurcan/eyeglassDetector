# Presentation Speech Notes

---

## Slide 1: Project Definition

Hello everyone. My project is called **AI-Powered Eyeglass Lens Edge Detection**.

The idea is simple: an optician takes a photo of a customer wearing glasses, and the app automatically finds the exact edges of the lenses. It shows where the lens meets the frame — think of it like a digital version of what an optician machine does.

The app uses a combination of deep learning and digital image processing — things like edge detection and contour analysis. And everything runs directly on the phone. No internet, no server, completely offline.

As you can see in the diagram, the flow is straightforward: the customer wears glasses, the optician takes a photo, the app detects the lens edges, and the optician sees the result on screen.

---

## Slide 2: Project Design

Now, how does it actually work inside?

The system has three main stages, all running on the smartphone.

**First**, a small and fast object detector called YOLOv8-Nano finds where the glasses are in the photo. It draws a bounding box around them.

**Second**, that bounding box is passed to MobileSAM — a lightweight version of Meta's Segment Anything Model. It creates a detailed mask of the frame and lens area.

**Third**, classical image processing takes over. Using techniques like Canny edge detection and contour analysis from OpenCV, we extract the precise inner edge of each lens from the mask.

Finally, the result is shown to the optician on the screen. The whole pipeline runs in a few seconds, entirely on the device.

---

## Slide 3: Project Requirements

For the software side, I'm using React Native for the mobile app, ONNX Runtime to run the AI models on the phone, and OpenCV for the image processing part. The two main models are YOLOv8-Nano for detection and MobileSAM for segmentation.

On the hardware side, I'm developing on a Mac with a GPU for training and testing. The target device is a modern smartphone — iPhone or Android — with camera access.

For data, I have a personal test dataset of over 40 eyeglass images, and I'm also using public datasets from Roboflow and Kaggle for training.

---

## Slide 4: Success Criteria

I have three clear success criteria.

**First — Lens Detection accuracy of at least 85%.** If I give the app 100 photos of people wearing glasses, it should correctly detect and outline the lenses in at least 85 of them. I will test this by running the full pipeline on my test set and counting the successes.

**Second — Latency under 3 seconds.** The entire process — detection, segmentation, and edge refinement — must finish in under 3 seconds on a mid-range phone like an iPhone 13. I will measure this on a real device.

**Third — Works fully offline.** The app must work without any internet connection. An optician should be able to use it anywhere — even in a shop with no Wi-Fi. I will test this by turning on airplane mode and running the full pipeline.

---

## Slide 5: Project Timeline

The project is planned for 6 months.

In **Month 1**, I will research and compare models — MobileSAM versus SAM 2 Tiny — and pick the best one for mobile.

In **Month 2**, I will build the full pipeline on desktop using Python, simulating what will later run on the phone.

In **Month 3**, I will convert the models to mobile format — ONNX export, quantization, and testing the size and speed.

In **Month 4**, I will start building the actual mobile app with React Native and connect the camera.

In **Month 5**, I will integrate everything — connect the detector, the segmentation model, and OpenCV together on the phone.

And in **Month 6**, I will focus on performance tuning, fixing edge cases, and polishing the user interface for release.

---

## Slide 6: References

My main references include the SAM 2 paper from Meta, the MobileSAM paper, ONNX Runtime documentation, the original Canny edge detection paper, and the React Native framework documentation.

Thank you. I'm happy to take any questions.
