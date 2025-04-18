import torch
import torch.nn as nn
import torch.nn.functional as F

class BoneFractureDetector(nn.Module):
    def __init__(self, num_classes=7, num_points=4):  # 4 points => 8 coordonnées (x, y)
        super(BoneFractureDetector, self).__init__()
        
        # Feature Extractor (CNN)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calcul automatique de la taille après convolutions
        self.flatten_size = 128 * (224 // 8) * (224 // 8)  # 224/2 -> 112 -> 56 -> 28

        # Branche Classification (Softmax sur 7 classes)
        self.fc_class = nn.Linear(self.flatten_size, 128)
        self.class_out = nn.Linear(128, num_classes)  # 7 classes
        
        # Branche Détection (Polygone avec 4 points)
        self.fc_bbox = nn.Linear(self.flatten_size, 128)
        self.bbox_out = nn.Linear(128, num_points * 2)  # (x1, y1, x2, y2, x3, y3, x4, y4)

        # Dropout pour éviter l'overfitting
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Passage dans les couches convolutionnelles
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten pour passer aux couches Fully Connected
        x = torch.flatten(x, 1)

        # Branche Classification
        class_pred = F.relu(self.fc_class(x))
        class_pred = self.dropout(class_pred)
        class_pred = self.class_out(class_pred)  # Pas de Softmax ici (inclus dans CrossEntropyLoss)

        # Branche Localisation (Polygone)
        bbox_pred = F.relu(self.fc_bbox(x))
        bbox_pred = self.bbox_out(bbox_pred)  # Pas d'activation car valeurs continues

        return class_pred, bbox_pred
