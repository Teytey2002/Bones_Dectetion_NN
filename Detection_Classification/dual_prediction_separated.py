import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from Detection_Classification.data_loader_detection_classification import test_dataset
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Classification.model import BoneFractureCNN
from Detection.polygon_detection_model import PolygonDetector

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger modèle classification
model_class = BoneFractureCNN(num_classes=7).to(device)
model_class.load_state_dict(torch.load("models/Classification/best_model_classification85.pth", map_location=device))
model_class.eval()

# Charger modèle détection
model_det = PolygonDetector(num_points=4).to(device)
model_det.load_state_dict(torch.load("models/Detection/polygon_detector_ReduceLROnPlateau_2.pth", map_location=device))
model_det.eval()

# Liste des classes
class_names = ['elbow positive', 'fingers positive', 'forearm fracture',
               'humerus fracture', 'humerus', 'shoulder fracture', 'wrist positive']

# Affichage comparatif
def show_comparison(image, label, class_pred, bbox_pred):
    image_np = image.permute(1, 2, 0).numpy()
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    width, height = image_np.shape[1], image_np.shape[0]
    true_class = int(label[0])
    true_coords = label[1:].reshape(-1, 2).cpu().numpy() * [width, height]
    pred_coords = bbox_pred.reshape(-1, 2) * [width, height]

    # Réel
    axs[0].imshow(image_np)
    axs[0].set_title(f"Réel: {class_names[true_class]}")
    true_poly = patches.Polygon(true_coords, edgecolor='r', facecolor='none', linewidth=2)
    axs[0].add_patch(true_poly)
    axs[0].axis("off")

    # Prédit
    axs[1].imshow(image_np)
    axs[1].set_title(f"Prédit: {class_names[class_pred]}")
    pred_poly = patches.Polygon(pred_coords, edgecolor='g', facecolor='none', linewidth=2)
    axs[1].add_patch(pred_poly)
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()

# Test
for i in range(5):
    image, label = test_dataset[i]
    image_input = image.unsqueeze(0).to(device)

    with torch.no_grad():
        # Prédiction de la classe
        class_output = model_class(image_input)
        class_pred = torch.argmax(class_output, dim=1).item()

        # Prédiction des coordonnées
        bbox_pred = model_det(image_input).squeeze().cpu().numpy()

    show_comparison(image, label, class_pred, bbox_pred)
