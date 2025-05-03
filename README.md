
# 🦴 Bone Fracture Detection in X-Ray Images using CNNs and Polygon Localization

This project explores deep learning approaches to automatically detect and localize bone fractures in upper limb X-ray images. It compares classification-only, polygon-based detection, multitask models, and YOLOv8-based methods.

## 📌 Objectives

- Classify the type of fracture (7 classes).
- Precisely localize fractures using polygon predictions.
- Compare multiple deep learning strategies:
  - Separate classification and detection models.
  - A multitask CNN model for joint classification + detection.
  - YOLOv8 object detectors.

## 📁 Project Structure

```
Bones_Dectetion_NN/
│
├── Classification/
│   ├── train.py                     # CNN training for classification
│   ├── test.py                      # Classification test script
│   ├── model.py                     # CNN model architecture
│   ├── data_loader.py              # Dataloader for classification
│   ├── train_loop.py                # Continuous training with early stopping
│   └── test_all.py                  # Evaluate all models + confusion matrix
│
├── Detection/
│   ├── train_polygon_detector.py    # Polygon detector training
│   ├── test_polygon_detector.py     # Visualize predictions
│   ├── polygon_detection_model.py   # CNN for polygon regression
│   ├── data_loader_detection.py     # Dataloader for detection
│   └── test_detection.py            # Detection metrics (MSE, avg. distance)
│
├── Detection_Classification/
│   ├── train_detection_classification.py     # Multitask model training
│   ├── test_detection_classification.py      # Multitask model evaluation
│   ├── model_detection_classification.py     # Multitask CNN architecture
│   └── data_loader_detection_classification.py
│
├── Dual/
│   └── dual_prediction_separated.py          # Separate model dual prediction
│
├── utils/
│   └── test_bounding_boxes.py       # Label visualization helper
│
├── models/                          # Saved models
├── runs*/                           # TensorBoard logs
├── Bones_Detection_Report.pdf       # Full project report (PDF)
```

## 🧠 Models

### 1. Classification-Only
- Custom CNN with 3 conv layers + dropout.
- Best test accuracy: **85.54%**
- TensorBoard visualization and loop training (`train_loop.py`)

### 2. Polygon Detection
- CNN predicting 4-point polygons (8 values).
- Loss: Smooth L1 (Huber)
- Results: MSE ≈ 0.0259, Avg. point distance ≈ 0.127

### 3. Multitask Model (classification + detection)
- 10-layer CNN (VGG-inspired)
- Classification head + detection head
- Accuracy: **68.21%**, Smooth L1 ≈ 0.015 per point

### 4. YOLOv8 Experiments
- YOLOv8n (nano) and YOLOv8s (small) evaluated.
- Underperformed compared to custom models:
  - YOLOv8n mAP@0.5 = 19.9%
  - YOLOv8s mAP@0.5 = 13.5%

## 🖼️ Dataset

- 📦 [Bone Fracture Detection - Kaggle](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project)
- Annotated with:
  - Class index (0–6)
  - Polygon coordinates (YOLO-style)
- Split: train / valid / test

## 🛠️ Installation

```bash
git clone https://github.com/Teytey2002/Bones_Dectetion_NN.git
cd Bones_Dectetion_NN

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Example `requirements.txt`:

```
torch
torchvision
numpy
matplotlib
scikit-learn
tensorboard
tqdm
Pillow
```

## ▶️ Usage

### Test classifier model:

```bash
cd Classification
python test.py
```

### Test polygon detector:

```bash
cd Detection
python test_polygon_detector.py
```

### Evaluate multitask model:

```bash
cd Detection_Classification
python test_detection_classification.py
```

### Visualize logs:

```bash
tensorboard --logdir runs/
```

## 📈 Results

| Method                | Accuracy | Localization Quality     | Notes                             |
|-----------------------|----------|---------------------------|------------------------------------|
| Classification-only   | 85.54%   | ❌                        | Best for classification            |
| Detection-only        | ❌       | Avg. dist = 0.127         | Precise polygon predictions        |
| Multitask Model       | 68.21%   | SmoothL1 ≈ 0.015          | Joint classification + detection   |
| YOLOv8n               | -        | mAP@0.5 = 19.9%           | Lightweight, low accuracy          |

## 📄 Report

A full scientific report of this project is available here: [`Bones_Detection_Report.pdf`](Bones_Detection_Report.pdf)

## 📚 References

- [Bone Fracture Detection Dataset - Kaggle](https://www.kaggle.com/datasets/pkdarabi/bone-fracture-detection-computer-vision-project)
- Simonyan & Zisserman (2015) — Very Deep Conv Nets
- Lundervold & Lundervold (2020) — Deep Learning in Medical Imaging

---

