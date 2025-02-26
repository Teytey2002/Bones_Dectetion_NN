import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image



# Définition des transformations
image_size = 224  # Taille standard pour les CNN
batch_size = 32   # Taille des lots

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalisation
])

# Chemin du dataset
data_dir_train = r'C:\Users\theod\OneDrive\Documents\ULB\Ma1\Project_Bones\DataSet\BoneFractureYolo8\train'
data_dir_valid = r'C:\Users\theod\OneDrive\Documents\ULB\Ma1\Project_Bones\DataSet\BoneFractureYolo8\valid'
data_dir_test = r'C:\Users\theod\OneDrive\Documents\ULB\Ma1\Project_Bones\DataSet\BoneFractureYolo8\test'


class BoneFractureClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")
        self.transform = transform

        # Filtrer les images qui ont un label existant et non vide
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
        image = Image.open(img_name).convert("RGB")  # Convertir en RGB (3 canaux)

        # Charger le fichier label correspondant
        label_name = os.path.join(self.label_dir, self.image_filenames[idx].replace(".jpg", ".txt"))
        with open(label_name, "r") as f:
            label_data = f.readline().strip().split()
            class_label = int(label_data[0])  # Extraire uniquement la classe
        
        label_tensor = torch.tensor(class_label, dtype=torch.long)

        # Appliquer les transformations
        if self.transform:
            image = self.transform(image)

        return image, label_tensor

# Création du dataset et DataLoader
train_dataset = BoneFractureClassificationDataset(root_dir=data_dir_train, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = BoneFractureClassificationDataset(root_dir=data_dir_valid, transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

test_dataset = BoneFractureClassificationDataset(root_dir=data_dir_test, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Vérification des données
dataiter = iter(train_loader)
images, labels = next(dataiter)

print("Shape des images:", images.shape)  # Doit être [batch_size, 3, 224, 224]
print("Labels associés:", labels)

# Vérification des données
dataiter = iter(valid_loader)
images, labels = next(dataiter)

print("Shape des images:", images.shape)  # Doit être [batch_size, 3, 224, 224]
print("Labels associés:", labels)

# Vérification des données
dataiter = iter(test_loader)
images, labels = next(dataiter)

print("Shape des images:", images.shape)  # Doit être [batch_size, 3, 224, 224]
print("Labels associés:", labels)