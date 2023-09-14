# Alertness-Detection-using-Yolov5

# YOLOv5 Real-time Object Detection

This comprehensive guide provides detailed technical instructions for setting up and utilizing YOLOv5 (You Only Look Once) for real-time object detection. YOLOv5 is a state-of-the-art deep learning framework known for its exceptional speed and accuracy in object detection tasks. Here, we will cover the entire process, including installation, real-time detection with a webcam feed, training a custom YOLOv5 model, and loading a custom model for object detection.

## Getting Started

### 1. Installation

To get started, ensure you have Python 3.10.5 installed. Then, install the required libraries and clone the YOLOv5 repository from GitHub:

```bash
pip3 install torch torchvision torchaudio
git clone https://github.com/ultralytics/yolov5
```

This will set up the necessary Python environment and obtain the YOLOv5 source code for your local development.

### 2. Real-time Detections

#### Script Overview

In this section, we'll employ a Python script for real-time object detection using a webcam feed. The script leverages the YOLOv5 model for inference on each frame captured from the webcam.

#### Script Execution

Here is the script to perform real-time detections:

```python
import torch
import numpy as np
import cv2

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize the webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    # Make detections
    results = model(frame)

    cv2.imshow('YOLO', np.squeeze(results.render()))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

Ensure your webcam is functioning correctly and press 'q' to exit the detection loop. Feel free to customize the script and adjust parameters to meet your specific requirements.

### 3. Training a Custom YOLOv5 Model (Optional)

#### Data Collection

If you intend to train a custom YOLOv5 model, you must collect a labeled dataset of images and corresponding annotations. This dataset should encompass the objects of interest, each annotated with bounding boxes.

#### Training Script

The YOLOv5 repository offers a training script (`train.py`) that you can employ for custom model training. The script accommodates various parameters, including image size, batch size, and the number of training epochs. Modify these parameters according to your dataset size and available hardware resources.

### 4. Loading a Custom Model

#### Loading Pre-trained Model Weights

If you have trained a custom YOLOv5 model, you can load it to perform real-time detections. Ensure you have the file path to your trained model weights.

```python
import torch
import numpy as np
import cv2

# Load the custom YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='/path/to/your/custom_model.pt', force_reload=True)

# Initialize the webcam feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    # Make detections
    results = model(frame)

    cv2.imshow('YOLO', np.squeeze(results.render()))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

Replace `/path/to/your/custom_model.pt` with the actual path to your custom-trained model weights.

## Dependencies

- Python 3.10.5
- torch 1.12.1
- [YOLOv5](https://github.com/ultralytics/yolov5)


Pictures/Video Below:
![Screen Shot 2022-10-03 at 4 25 40 PM](https://user-images.githubusercontent.com/108766004/193704501-51e3ee96-7cfa-454d-a93f-81e714e6fa48.png)
![Screen Shot 2022-10-03 at 4 33 19 PM](https://user-images.githubusercontent.com/108766004/193704509-b276f5ba-0a4e-4018-aaff-aee907ab9ac0.png)


https://user-images.githubusercontent.com/108766004/193704536-c0c6fe7e-e0f3-4231-8347-0c7c8381a6ae.mov

