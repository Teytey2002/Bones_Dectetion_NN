import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Vérifier si un GPU est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilisation de : {device}")

# Définition des transformations
image_size = 224  # Taille standard pour les modèles CNN
batch_size = 32   # Taille des lots

# Transforms pour chaque dataset
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(p=0.5),  # Data Augmentation (seulement sur train)
    transforms.RandomRotation(10),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalisation pour améliorer la convergence
])

test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Chargement des datasets depuis les dossiers
data_dir = r'C:\Users\theod\OneDrive\Documents\ULB\Ma1\Project_Bones\DataSet\BoneFractureYolo8'  # Change ce chemin avec l'emplacement de tes données

train_dataset = datasets.ImageFolder(root=f"{data_dir}/train", transform=train_transform)
valid_dataset = datasets.ImageFolder(root=f"{data_dir}/valid", transform=test_transform)
test_dataset = datasets.ImageFolder(root=f"{data_dir}/test", transform=test_transform)

# Création des DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Vérification du nombre d'images par classe
print(f"Classes trouvées : {train_dataset.classes}")
print(f"Nombre d'images - Train: {len(train_dataset)}, Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")

# Affichage de quelques images pour vérifier
def imshow(img):
    img = img / 2 + 0.5  # Dé-normalisation
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="gray")
    plt.axis("off")
    plt.show()

# Récupérer un batch d'images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Affichage des premières images
imshow(torchvision.utils.make_grid(images[:8]))  # Affiche 8 images

