import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO
import os
from PIL import Image

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger le modèle YOLOv8
model = YOLO(r'runs/detect/yolo_bones_v8n4/weights/best.pt')

# Liste des classes (tu peux récupérer automatiquement sinon depuis model.names)
class_names = ['elbow positive', 'fingers positive', 'forearm fracture',
               'humerus fracture', 'humerus', 'shoulder fracture', 'wrist positive']

# Fonctions auxiliaires
def load_ground_truth(label_path):
    with open(label_path, "r") as f:
        lines = f.readlines()
        if not lines or lines[0].strip() == '':
            # Aucun label valide
            return None, None
        line = lines[0].strip().split()
        class_label = int(line[0])
        coords = list(map(float, line[1:]))
    return class_label, coords


def show_comparison(image_path, label_path, prediction):
    # Charger image
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    image_np = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0

    # Charger Ground Truth
    true_class, true_coords = load_ground_truth(label_path)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Partie réelle
    axs[0].imshow(image)
    if true_class is not None and true_coords is not None:
        true_coords = np.array(true_coords).reshape(-1, 2) * [width, height]
        true_poly = patches.Polygon(true_coords, edgecolor='red', facecolor='none', linewidth=2)
        axs[0].add_patch(true_poly)
        axs[0].set_title(f"Réel: {class_names[true_class]}")
    else:
        axs[0].set_title("Réel: Aucun label")
    axs[0].axis("off")

    # Partie prédite
    axs[1].imshow(image)
    if prediction.boxes is not None and len(prediction.boxes) > 0:
        pred_boxes = prediction.boxes.xyxy.cpu().numpy()
        pred_classes = prediction.boxes.cls.cpu().numpy()
        x1, y1, x2, y2 = pred_boxes[0]  # Prend la première bbox
        class_pred = int(pred_classes[0])
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, edgecolor='lime', facecolor='none', linewidth=2)
        axs[1].add_patch(rect)
        axs[1].set_title(f"Prédit: {class_names[class_pred]}")
    else:
        axs[1].set_title("Prédit: Rien détecté")
    axs[1].axis("off")

    plt.tight_layout()
    plt.show()


# --- SCRIPT PRINCIPAL ---
import numpy as np

# Dossiers test
test_images_dir = r'C:\Users\theod\OneDrive\Documents\ULB\Ma1\Project_Bones\DataSet\BoneFractureYolo8\test\images'
test_labels_dir = r'C:\Users\theod\OneDrive\Documents\ULB\Ma1\Project_Bones\DataSet\BoneFractureYolo8\test\labels'

# Prédire sur toutes les images du dossier test
results = model.predict(source=test_images_dir, save=False, imgsz=640, conf=0.25, device=0)

# Boucle pour afficher 5 exemples
for i, result in enumerate(results):
    image_path = result.path
    filename = os.path.basename(image_path)
    label_path = os.path.join(test_labels_dir, filename.replace(".jpg", ".txt"))

    if os.path.exists(label_path):
        show_comparison(image_path, label_path, result)

    if i >= 4:  # Limite à 5 images
        break
