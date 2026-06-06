# WS-Object-detection-using-web-camera

## NAME:KATHI HASINI
## REG NO:212224240074

## AIM:
To perform real-time object detection using a trained YOLO v4 model through your laptop camera.

## Algorithm:
Initialize the webcam using OpenCV. Load the YOLOv4 pre-trained model (yolov4.weights) and configuration (yolov4.cfg). Load the COCO class labels (coco.names) and assign random colors for visualization. Capture frames continuously from the webcam. Preprocess each frame by creating a blob and pass it through the YOLOv4 network. Extract detected object bounding boxes, class IDs, and confidences. Apply Non-Maximum Suppression (NMS) to remove overlapping boxes. Draw bounding boxes and labels for detected objects on the frame. Display the frames inline in Jupyter Notebook using matplotlib. Stop detection by pressing the “Stop Detection” button in the notebook


## PROGRAM:
```
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Read image
image = cv2.imread("8765.jpg")

# Perform object detection
results = model(image)

# Print detected objects
for box in results[0].boxes:
    cls = int(box.cls)
    conf = float(box.conf)
    print(f"{model.names[cls]} : {conf:.2f}")

# Draw detections on image
output = results[0].plot()

# Convert BGR to RGB
output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

# Display image
plt.figure(figsize=(12, 8))
plt.imshow(output)
plt.axis('off')
plt.show()

```

## OUTPUT:
#### ORIGINAL IMAGE:
<img width="668" height="358" alt="image" src="https://github.com/user-attachments/assets/223c87ed-8616-4647-9746-04de036ce0a3" />


#### OBJECT DETECTED IMAGE:

<img width="858" height="458" alt="image" src="https://github.com/user-attachments/assets/6656fae8-45b4-4618-8ee3-57743abe1dc4" />


## RESULT:
The webcam captures live video frames. YOLOv4 detects objects like person, chair, laptop, bottle, etc. in real-time. Detected objects are highlighted with bounding boxes and class labels. Frames are displayed inline in the Jupyter Notebook. Detection stops when the “Stop Detection” button is pressed.
