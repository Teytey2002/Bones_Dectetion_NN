import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from data_loader_detection_classification import test_dataset
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Classification.model import BoneFractureCNN
from Detection.polygon_detection_model import PolygonDetector
from Classification.data_loader import test_dataset as test_dataset_classification  # 224x224
from Detection.data_loader_detection import test_dataset as test_dataset_detection  # 640x640

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger modèle classification
model_class = BoneFractureCNN(num_classes=7).to(device)
model_class.load_state_dict(torch.load("models/Classification/best_model_classification85.pth", map_location=device))
model_class.eval()

# Charger modèle détection
model_det = PolygonDetector(num_points=4).to(device)
model_det.load_state_dict(torch.load("models/Detection/loop/polygon_detector_best_model.pth", map_location=device))
model_det.eval()

# Liste des classes
class_names = ['elbow positive', 'fingers positive', 'forearm fracture',
               'humerus fracture', 'humerus', 'shoulder fracture', 'wrist positive']

# Affichage comparatif
def show_comparison(image, label, class_pred, bbox_pred):
    image_np = image.permute(1, 2, 0).numpy()

    image_np = (image_np * 0.5) + 0.5
    image_np = np.clip(image_np, 0, 1)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    width, height = image_np.shape[1], image_np.shape[0]
    true_class = int(label[0])
    true_coords = label[1:].reshape(-1, 2).cpu().numpy() * [width, height]
    pred_coords = bbox_pred.reshape(-1, 2) * [width, height]

    # Réel
    axs[0].imshow(image_np)
    axs[0].set_title(f"True: {class_names[true_class]}")
    true_poly = patches.Polygon(true_coords, edgecolor='r', facecolor='none', linewidth=2)
    axs[0].add_patch(true_poly)
    axs[0].axis("off")

    # Prédit
    axs[1].imshow(image_np)
    axs[1].set_title(f"Predict: {class_names[class_pred]}")
    pred_poly = patches.Polygon(pred_coords, edgecolor='g', facecolor='none', linewidth=2)
    axs[1].add_patch(pred_poly)
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()


for i in range(15):
    # Prendre la même image
    img_class, label_class = test_dataset_classification[i]
    img_det, label_det = test_dataset_detection[i]

    image_input_class = img_class.unsqueeze(0).to(device)
    image_input_det = img_det.unsqueeze(0).to(device)

    with torch.no_grad():
        # Prédiction de la classe
        class_output = model_class(image_input_class)
        class_pred = torch.argmax(class_output, dim=1).item()

        # Prédiction des coordonnées
        bbox_pred = model_det(image_input_det).squeeze().cpu().numpy()

    # Attention on utilise l'image de détection pour l'affichage
    show_comparison(img_det, label_det, class_pred, bbox_pred)