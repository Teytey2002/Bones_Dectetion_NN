import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

image_size = 224
batch_size = 32

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor()
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

        return image, label_tensor

train_dataset = PolygonDetectionDataset(root_dir=data_dir_train, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

valid_dataset = PolygonDetectionDataset(root_dir=data_dir_valid, transform=transform)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

test_dataset = PolygonDetectionDataset(root_dir=data_dir_test, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
