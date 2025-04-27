import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import random

class CustomTransform:
    def __init__(self, image_size=640):
        self.image_size = image_size
        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2)

    def __call__(self, image, coords):
        # Resize
        image = TF.resize(image, (self.image_size, self.image_size))

        # Random Rotation
        angle = random.uniform(-15, 15)
        image = TF.rotate(image, angle)
        coords = self.rotate_coords(coords, angle)

        # Random Horizontal Flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            coords[:, 0] = 1.0 - coords[:, 0]

        # Random Affine Translation
        max_translate = 0.1
        trans_x = random.uniform(-max_translate, max_translate)
        trans_y = random.uniform(-max_translate, max_translate)
        image = TF.affine(image, angle=0, translate=(int(trans_x * self.image_size), int(trans_y * self.image_size)), scale=1.0, shear=[0.0, 0.0])
        coords[:, 0] += trans_x
        coords[:, 1] += trans_y

        # Clamp coords
        coords = torch.clamp(coords, 0.0, 1.0)

        # Color Jitter
        image = self.color_jitter(image)

        # To Tensor
        image = TF.to_tensor(image)

        return image, coords

    def rotate_coords(self, coords, angle_deg):
        angle_rad = torch.tensor(-angle_deg * 3.14159265 / 180.0)
        center = torch.tensor([0.5, 0.5])
        rot_matrix = torch.tensor([
            [torch.cos(angle_rad), -torch.sin(angle_rad)],
            [torch.sin(angle_rad), torch.cos(angle_rad)]
        ])
        coords = coords - center
        coords = coords @ rot_matrix.T
        coords = coords + center
        return coords
    
class CustomTestTransform:
    def __init__(self, image_size=640):
        self.image_size = image_size

    def __call__(self, image, coords):
        image = TF.resize(image, (self.image_size, self.image_size))
        image = TF.to_tensor(image)
        return image, coords

