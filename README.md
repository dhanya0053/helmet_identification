Helmet Attribute Detection using YOLOv8

This project implements a helmet attribute detection system using YOLOv8
.
A custom dataset of 1000+ annotated images was created and labeled to capture detailed helmet attributes, including:

✅ Presence of helmet

✅ Big surface cracks on helmet

✅ Chin belt usage

✅ Helmet color

✅ Team identification (based on logo or design)

This model can be applied in manufacturing, refurbishing of helmets.

🚀 Features

Multi-class and multi-attribute helmet detection

Custom dataset with 1500+ annotated images

YOLOv8 training pipeline with Roboflow integration

Supports inference on images and can be extended to video streams

Detects damage, compliance, and identity in helmets

📂 Project Structure
helmet3.ipynb        # Jupyter Notebook with training + inference code
runs/detect/         # YOLOv8 training outputs (generated after training)
data.yaml            # Dataset configuration (Roboflow export)

🛠️ Installation

Clone the repository and install dependencies:

git clone https://github.com/dhanya0053/helmet-attribute-detection.git
cd helmet-attribute-detection
pip install -r requirements.txt


Dependencies:

Python 3.8+

ultralytics

roboflow

Or install manually:

pip install ultralytics roboflow

📊 Dataset

The dataset was created manually by annotating 1000+ helmet images with multiple attributes.
Exported via Roboflow in YOLOv8 format.
https://universe.roboflow.com/dhanya-c46vg/helmet_identification/browse
Attributes include:

Surface cracks (big cracks vs no cracks)

Chin belt (fastened/unfastened)

Helmet color (e.g., red, blue, yellow, white)

Team classification (logo/design-based)

Example annotation (YOLO format):

class_id x_center y_center width height

🏋️ Training

Training YOLOv8 model on the dataset:

from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # pretrained YOLOv8 nano model
model.train(
    data="helmet_identification-7/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    name="helmet_attribute_detector"
)

🔍 Inference

Run detection on new images:

model = YOLO("runs/detect/helmet_attribute_detector/weights/best.pt")
results = model("test.jpg", show=True)


You can also export the model for deployment:

yolo export model=runs/detect/helmet_attribute_detector/weights/best.pt format=onnx

📈 Results

Dataset: 1000+ custom annotated images

Model: YOLOv8n

Epochs: 50


Evaluation Metrics (Class B sample results)
- Precision: 61.17%
- Recall: 56.84%
- mAP@50: 59.95%
- mAP@50-95: 44.68%
- Fitness Score:0.4621

**Speed (per image):**
- Preprocessing: 0.22 ms  
- Inference: 3.00 ms  
- Postprocessing: 3.40 ms  
➡️ **~6.6 ms total (~150 FPS)**
<img width="257" height="196" alt="image" src="https://github.com/user-attachments/assets/e4258d36-2888-4796-9d2e-323c08dc1c14" />


📌 Future Improvements

Real-time helmet attribute detection from live video

Support for more helmet attributes (material type, sticker detection)

Deployment on mobile/edge devices (Jetson Nano, Raspberry Pi)
