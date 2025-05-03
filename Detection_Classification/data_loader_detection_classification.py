import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Add for test bounding boxes on images
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as TF

# Définition des transformations (Data Augmentation + Normalisation)
image_size = 224
batch_size = 32   

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Data Augmentation
    #transforms.RandomRotation(10),  # Rotation légère
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])

test_transform = transforms.Compose([   # Because no augmentation for test and validation
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Chemin du dataset
data_dir_train = r'C:\Users\theod\OneDrive\Documents\ULB\Ma1\Project_Bones\DataSet\BoneFractureYolo8\train'
data_dir_valid = r'C:\Users\theod\OneDrive\Documents\ULB\Ma1\Project_Bones\DataSet\BoneFractureYolo8\valid'
data_dir_test = r'C:\Users\theod\OneDrive\Documents\ULB\Ma1\Project_Bones\DataSet\BoneFractureYolo8\test'

class BoneFractureDetectionDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_points=4):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")
        self.transform = transform
        self.num_points = num_points  # Nombre de points normalisés
        
        # Liste des images avec labels existants et non vides
        self.image_filenames = []
        for img_file in sorted(os.listdir(self.image_dir)):
            label_file = os.path.join(self.label_dir, img_file.replace(".jpg", ".txt"))
            if os.path.exists(label_file) and os.path.getsize(label_file) > 0:
                self.image_filenames.append(img_file)

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Charger l'image
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_name).convert("RGB")  
        
        # Charger le fichier label correspondant
        label_name = os.path.join(self.label_dir, self.image_filenames[idx].replace(".jpg", ".txt"))
        with open(label_name, "r") as f:
            label_data = f.readline().strip().split()
        
        # Extraction des valeurs (classe + points x,y normalisés)
        class_label = int(label_data[0])  
        coords = [float(x) for x in label_data[1:]]  

        # Normalisation des coordonnées à toujours 4 points (8 valeurs)
        while len(coords) < 2 * self.num_points:  
            coords.append(0.0)  # Complète avec des zéros
        coords = coords[:2 * self.num_points]  # Tronque si trop de points

        label_tensor = torch.tensor([class_label] + coords, dtype=torch.float32)

        # Appliquer les transformations
        if self.transform:
            image = self.transform(image)

        return image, label_tensor

# Création du dataset et DataLoader
train_dataset = BoneFractureDetectionDataset(root_dir=data_dir_train, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = BoneFractureDetectionDataset(root_dir=data_dir_valid, transform=test_transform)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

test_dataset = BoneFractureDetectionDataset(root_dir=data_dir_test, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)






