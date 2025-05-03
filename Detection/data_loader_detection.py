import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
#from custom_transform_augmented import CustomTransform, CustomTestTransform
import matplotlib.pyplot as plt
import numpy as np
image_size = 640   # Before was 224
batch_size = 32    # Before was 32



#transform = CustomTransform(image_size=640)
#test_transform = CustomTestTransform(image_size=640)
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    #transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Data Augmentation
    #transforms.RandomRotation(10),  # Rotation légère
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  
])

test_transform = transforms.Compose([   # Because no augmentation for test and validation
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

data_dir_train = r'C:\Users\theod\OneDrive\Documents\ULB\Ma1\Project_Bones\DataSet\BoneFractureYolo8\train'
data_dir_valid = r'C:\Users\theod\OneDrive\Documents\ULB\Ma1\Project_Bones\DataSet\BoneFractureYolo8\valid'
data_dir_test = r'C:\Users\theod\OneDrive\Documents\ULB\Ma1\Project_Bones\DataSet\BoneFractureYolo8\test'

class PolygonDetectionDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_points=4):
        self.image_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")
        self.transform = transform
        self.num_points = num_points

        self.image_filenames = [f for f in sorted(os.listdir(self.image_dir))
                                if os.path.exists(os.path.join(self.label_dir, f.replace(".jpg", ".txt")))
                                and os.path.getsize(os.path.join(self.label_dir, f.replace(".jpg", ".txt"))) > 0]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_filenames[idx])
        image = Image.open(image_path).convert("RGB")

        label_path = os.path.join(self.label_dir, self.image_filenames[idx].replace(".jpg", ".txt"))
        with open(label_path, "r") as f:
            data = f.readline().strip().split()

        coords = [float(x) for x in data[1:]]
        while len(coords) < 2 * self.num_points:
            coords.append(0.0)
        coords = coords[:2 * self.num_points]
        label_tensor = torch.tensor([int(data[0])] + coords, dtype=torch.float32)

        if self.transform: 
            image = self.transform(image)

        #if self.transform:     #For data augmentation
        #    coords_tensor = torch.tensor(coords, dtype=torch.float32).reshape(-1, 2)
        #    image, coords_tensor = self.transform(image, coords_tensor)
        #    coords = coords_tensor.view(-1).tolist()


        return image, label_tensor

train_dataset = PolygonDetectionDataset(root_dir=data_dir_train, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = PolygonDetectionDataset(root_dir=data_dir_valid, transform=test_transform)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

test_dataset = PolygonDetectionDataset(root_dir=data_dir_test, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

## Print images of test dataset to see results 
#if __name__ == "__main__":
#   
#    for i in range(15):
#        image, label = test_dataset[i]
#        image = image.permute(1, 2, 0).numpy()
#        
#        # Unnormalize pour afficher correctement
#        image = (image * 0.5) + 0.5
#        image = np.clip(image, 0, 1)
#        plt.imshow(image)
#        plt.title(f"Label: {label}")
#        plt.show()