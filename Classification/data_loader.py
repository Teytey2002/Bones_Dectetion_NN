import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Définition des paramètres
image_size = 224  # Taille standard pour les CNN
batch_size = 32   # Taille des lots

# Chemins du dataset
data_dir_train = r'C:\Users\theod\OneDrive\Documents\ULB\Ma1\Project_Bones\DataSet\BoneFractureYolo8\train'
data_dir_valid = r'C:\Users\theod\OneDrive\Documents\ULB\Ma1\Project_Bones\DataSet\BoneFractureYolo8\valid'
data_dir_test = r'C:\Users\theod\OneDrive\Documents\ULB\Ma1\Project_Bones\DataSet\BoneFractureYolo8\test'

# Transformations pour l'entraînement (avec Data Augmentation)
train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(p=0.5),  # Flip horizontal (50% des cas)
    transforms.RandomRotation(10),  # Rotation aléatoire entre -10° et +10°
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Petits déplacements
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Distorsion légère
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Variations de couleur
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Transformations pour validation et test (sans Data Augmentation)
test_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Définition du Dataset personnalisé
class BoneFractureClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")
        self.transform = transform

        # Liste des fichiers images ayant un label valide
        self.image_filenames = [
            img_file for img_file in sorted(os.listdir(self.image_dir))
            if os.path.exists(os.path.join(self.label_dir, img_file.replace(".jpg", ".txt"))) 
            and os.path.getsize(os.path.join(self.label_dir, img_file.replace(".jpg", ".txt"))) > 0
        ]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Charger l'image
        img_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert("RGB")

        # Charger le label correspondant
        label_path = os.path.join(self.label_dir, self.image_filenames[idx].replace(".jpg", ".txt"))
        with open(label_path, "r") as f:
            label_data = f.readline().strip().split()
            class_label = int(label_data[0])  # Extraire uniquement la classe
        
        label_tensor = torch.tensor(class_label, dtype=torch.long)

        # Appliquer les transformations
        if self.transform:
            image = self.transform(image)

        return image, label_tensor

# Création des datasets et DataLoaders
train_dataset = BoneFractureClassificationDataset(root_dir=data_dir_train, transform=train_transform)
valid_dataset = BoneFractureClassificationDataset(root_dir=data_dir_valid, transform=test_transform)
test_dataset = BoneFractureClassificationDataset(root_dir=data_dir_test, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
