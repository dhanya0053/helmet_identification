# 🪖 Helmet Attribute Detection using YOLOv8

This project implements a **helmet attribute detection system** using **YOLOv8**.
A custom dataset of **1000+ annotated images** was created and labeled to capture detailed helmet attributes, including:

* ✅ Presence of helmet
* ✅ Big surface cracks on helmet
* ✅ Chin belt usage
* ✅ Helmet color
* ✅ Team identification (based on logo or design)

This model can be applied in **manufacturing** and **refurbishing of helmets**.

---

## 🚀 Features

* Multi-class and multi-attribute helmet detection
* Custom dataset with **1500+ annotated images**
* YOLOv8 training pipeline with **Roboflow integration**
* Supports inference on **images** and extendable to **video streams**
* Detects **damage, compliance, and identity** in helmets

---

## 📂 Project Structure

```
helmet3.ipynb        # Jupyter Notebook with training + inference code
runs/detect/         # YOLOv8 training outputs (generated after training)
data.yaml            # Dataset configuration (Roboflow export)
```

---

## 🛠️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/dhanya0053/helmet-attribute-detection.git
cd helmet-attribute-detection
pip install -r requirements.txt
```

**Dependencies:**

* Python 3.8+
* ultralytics
* roboflow

Or install manually:

```bash
pip install ultralytics roboflow
```

---

## 📊 Dataset

The dataset was created manually by annotating **1000+ helmet images** with multiple attributes.
Exported via **Roboflow** in YOLOv8 format.

🔗 [Dataset on Roboflow](https://app.roboflow.com/dhanya-c46vg/helmet_identification/browse)

**Attributes include:**

* Surface cracks (big cracks vs. no cracks)
* Chin belt (fastened/unfastened)
* Helmet color (e.g., red, blue, yellow, white)
* Team classification (logo/design-based)

**Example annotation (YOLO format):**

```
class_id x_center y_center width height
```

---

## 🏋️ Training

Training YOLOv8 model on the dataset:

```python
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
```

---

## 🔍 Inference

Run detection on new images:

```python
from ultralytics import YOLO

model = YOLO("runs/detect/helmet_attribute_detector/weights/best.pt")
results = model("test.jpg", show=True)
```

Export the model for deployment:

```bash
yolo export model=runs/detect/helmet_attribute_detector/weights/best.pt format=onnx
```

---

## 📈 Results

* **Dataset:** 1000+ custom annotated images
* **Model:** YOLOv8n
* **Epochs:** 50
<img width="257" height="196" alt="image" src="https://github.com/user-attachments/assets/51db7da0-8172-4c00-b07e-a682d298c4c6" />

**Evaluation Metrics (Class B sample results):**

* Precision: **61.17%**
* Recall: **56.84%**
* mAP\@50: **59.95%**
* mAP\@50-95: **44.68%**
* Fitness Score: **0.4621**

**Speed (per image):**

* Preprocessing: 0.22 ms
* Inference: 3.00 ms
* Postprocessing: 3.40 ms
  ➡️ **\~6.6 ms total (\~150 FPS)**

*(Add sample detection images here)*

---

## 📌 Future Improvements

* Real-time helmet attribute detection from live video
* Support for more helmet attributes (e.g., material type, sticker detection)
* Deployment on mobile/edge devices (Jetson Nano, Raspberry Pi)

---

## 👩‍💻 Author

**Dhanya**
4th Year Computer Science Student | Deep Learning Enthusiast
