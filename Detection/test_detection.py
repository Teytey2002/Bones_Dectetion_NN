import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from model_detection import BoneFractureDetector  # Import du modèle
from data_loader_detection import test_dataset  # Import du dataset de test

# Vérifier si un GPU est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing on: {device}")

# Charger le modèle entraîné
model_path = "models/bone_fracture_detection.pth"
model = BoneFractureDetector(num_classes=7, num_points=4).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Liste des classes
class_names = ['elbow positive', 'fingers positive', 'forearm fracture',
               'humerus fracture', 'humerus', 'shoulder fracture', 'wrist positive']

# Transformation pour le test
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Fonction pour afficher une image avec ses labels
def show_comparison(image, label, pred_class, pred_bbox):
    image = image.permute(1, 2, 0).numpy()
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    width, height = image.shape[1], image.shape[0]
    real_class = int(label[0])
    # Assurez-vous que label[1:] est bien un tensor float et le convertir en numpy
    real_coords = label[1:].cpu().numpy().reshape(-1, 2) * np.array([width, height])
    pred_coords = pred_bbox.reshape(-1, 2) * [width, height]
    
    # Image avec annotations réelles
    axs[0].imshow(image)
    axs[0].set_title(f"Réel: {class_names[real_class]}")
    real_polygon = patches.Polygon(real_coords, linewidth=2, edgecolor='r', facecolor='none')
    axs[0].add_patch(real_polygon)
    axs[0].axis("off")
    
    # Image avec prédictions du modèle
    axs[1].imshow(image)
    axs[1].set_title(f"Prédit: {class_names[pred_class]}")
    pred_polygon = patches.Polygon(pred_coords, linewidth=2, edgecolor='g', facecolor='none')
    axs[1].add_patch(pred_polygon)
    axs[1].axis("off")
    
    plt.show()

# Tester quelques images
num_samples = 5
for i in range(num_samples):
    image, label = test_dataset[i]
    # Vérifier si l'image est déjà un tensor, sinon la convertir en PIL.Image
    if isinstance(image, torch.Tensor):
        image = transforms.ToPILImage()(image)  # Convertir en PIL.Image

    # Appliquer la transformation et ajouter une dimension batch
    image = transform(image).unsqueeze(0).to(device)  # Transformation et ajout d'une dimension batch
    
    with torch.no_grad():
        class_pred, bbox_pred = model(image)
        class_pred = torch.argmax(class_pred, dim=1).item()
        bbox_pred = bbox_pred.squeeze().cpu().numpy()
    
    # Afficher la comparaison entre réel et prédiction
    show_comparison(image.squeeze().cpu(), label, class_pred, bbox_pred)
