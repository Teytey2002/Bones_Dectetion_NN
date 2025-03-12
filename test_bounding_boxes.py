import os
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from PIL import Image

# Chemin du dataset (à modifier si besoin)
data_dir = r'C:\Users\theod\OneDrive\Documents\ULB\Ma1\Project_Bones\DataSet\BoneFractureYolo8\train'  
image_dir = os.path.join(data_dir, "images")
label_dir = os.path.join(data_dir, "labels")

# Sélection d'une image et son label
image_name = "image2_25_png.rf.155e7a053672c6b22cf2bbe27611e3de.jpg"  # Mets ici une image de ton dataset
label_name = image_name.replace(".jpg", ".txt")

image_path = os.path.join(image_dir, image_name)
label_path = os.path.join(label_dir, label_name)

# Chargement de l'image
image = Image.open(image_path).convert("RGB")
width, height = image.size  # Taille originale de l'image

# Lecture du fichier label
with open(label_path, "r") as f:
    label_data = f.readline().strip().split()  # Lire la première ligne et la séparer

# Extraction des coordonnées du polygone
if len(label_data) > 1:
    class_id = int(label_data[0])  # Classe (ex: 2 = "forearm fracture")
    coords = [float(x) for x in label_data[1:]]  # Coordonnées normalisées

    # Convertir les coordonnées normalisées en pixels
    polygon_points = [(coords[i] * width, coords[i+1] * height) for i in range(0, len(coords), 2)]

    # Création de la figure
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Ajout du polygone
    polygon = patches.Polygon(polygon_points, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(polygon)

    # Ajouter le texte de la classe
    class_names = ['elbow positive', 'fingers positive', 'forearm fracture', 
                   'humerus fracture', 'humerus', 'shoulder fracture', 'wrist positive']
    ax.text(width - 10, 10, class_names[class_id], 
            color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5), ha='right', va='top')

    # Affichage
    plt.axis("off")
    plt.show()

else:
    print("Aucune annotation trouvée pour cette image.")