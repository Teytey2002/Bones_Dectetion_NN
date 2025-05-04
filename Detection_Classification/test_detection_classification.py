import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn as nn
import numpy as np
from data_loader_detection_classification import test_loader, test_dataset
from model_detection_classification import BoneFractureMultiTaskDeepNet
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger le modèle multitâche
model = BoneFractureMultiTaskDeepNet(num_classes=7, num_points=4).to(device)
model.load_state_dict(torch.load("models/Detection_Classification/bone_fracture_multitask_conv10_normalized_schedule_0.6dp.pth", map_location=device))
model.eval()

# Liste des classes
class_names = ['elbow positive', 'fingers positive', 'forearm fracture',
               'humerus fracture', 'humerus', 'shoulder fracture', 'wrist positive']

# Fonction pour affichage comparatif
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

# --- Partie 1 : Affichage de quelques images ---
for i in range(5):
    image, label = test_dataset[i]
    image_input = image.unsqueeze(0).to(device)

    with torch.no_grad():
        class_output, bbox_output = model(image_input)
        class_pred = torch.argmax(class_output, dim=1).item()
        bbox_pred = bbox_output.squeeze().cpu().numpy()

    show_comparison(image, label, class_pred, bbox_pred)

# --- Partie 2 : Évaluation sur tout le test_loader ---
all_mse = []
all_distances = []
correct = 0
total = 0

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluation Multitâche"):
        images, labels = images.to(device), labels.to(device)
        class_labels = labels[:, 0].long()
        true_coords = labels[:, 1:]

        class_preds, bbox_preds = model(images)

        # Classification
        _, predicted_classes = torch.max(class_preds, 1)
        correct += (predicted_classes == class_labels).sum().item()
        total += class_labels.size(0)

        # Détection : comparaison image par image
        for i in range(images.size(0)):
            true = true_coords[i].cpu().numpy()
            pred = bbox_preds[i].cpu().numpy()

            mse = mean_squared_error(true, pred)
            dist = np.mean(np.sqrt((np.array(true) - np.array(pred))**2))

            all_mse.append(mse)
            all_distances.append(dist)

# Résumé des métriques
avg_mse = np.mean(all_mse)
avg_dist = np.mean(all_distances)
accuracy = 100 * correct / total

print(f"\n----- ✅ Évaluation multitâche terminée -----")
print(f" - Classification Accuracy : {accuracy:.2f}%")
print(f" - MSE moyen des polygones : {avg_mse:.6f}")
print(f" - Distance moyenne entre points (normalisée) : {avg_dist:.6f}")
