import os
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Définition des transformations
image_size = 224  # Taille standard pour les CNN
batch_size = 32   # Taille des lots

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalisation
])

# Chemin du dataset
data_dir = r'C:\Users\theod\OneDrive\Documents\ULB\Ma1\Project_Bones\DataSet\BoneFractureYolo8\train'  # À modifier !

class BoneFractureDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_label_length=12):  # max_label_length défini à 12 (ajuster si besoin)
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")
        self.transform = transform
        self.image_filenames = sorted(os.listdir(self.image_dir))  # Trier pour correspondre aux labels
        self.max_label_length = max_label_length  # Longueur max des labels

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Charger l'image
        img_name = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(img_name).convert("RGB")  # Convertir en RGB (3 canaux)

        # Charger le fichier label correspondant
        label_name = os.path.join(self.label_dir, self.image_filenames[idx].replace(".jpg", ".txt"))
        if os.path.exists(label_name):
            with open(label_name, "r") as f:
                label_data = f.readline().strip().split()  # Lire la première ligne et la séparer
                if len(label_data) > 1:
                    label = [float(x) for x in label_data[1:]]  # Exclure le premier élément (classe)
                else:
                    label = []  # Cas où le fichier est vide
        else:
            label = []  # Aucun label trouvé

        # Remplissage avec des zéros si nécessaire
        label = label[:self.max_label_length]  # Tronquer si trop long
        label += [0.0] * (self.max_label_length - len(label))  # Compléter avec des zéros

        label_tensor = torch.tensor(label, dtype=torch.float32)

        # Appliquer les transformations
        if self.transform:
            image = self.transform(image)

        return image, label_tensor

# Création du dataset et DataLoader
train_dataset = BoneFractureDataset(root_dir=data_dir, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Vérification : Affichage des images
def imshow(img):
    img = img / 2 + 0.5  # Dé-normalisation
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap="gray")
    plt.axis("off")
    plt.show()

# Récupérer un batch d'images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Affichage des images
imshow(torchvision.utils.make_grid(images[:1]))  # Afficher 1 images
print("Labels associés : ", labels[:1])  # Afficher les labels
