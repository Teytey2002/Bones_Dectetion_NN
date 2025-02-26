import torch
import torch.nn as nn
import torch.nn.functional as F

class BoneFractureCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(BoneFractureCNN, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.dropout = nn.Dropout(0.5)  # Ajoute un Dropout à 50%
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32, eps=1e-5)  # Batch Normalization
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-5)  # Batch Normalization
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128, eps=1e-5)  # Batch Normalization

        #self.fc1 = nn.Linear(64 * 56 * 56, 128)  # Fully Connected with fixed size
        # Calculer dynamiquement la taille après convolutions
        self.flatten_size = self.compute_flatten_size()
        self.fc1 = nn.Linear(self.flatten_size, 128)    # Fully Connected with dynamic size
        self.fc2 = nn.Linear(128, num_classes)  # Sortie

  
    
    # Ajout de cette methode car on a ajoute une couche de convolution et du coup fc1 n'est plus correct
    def compute_flatten_size(self, input_shape=(3, 224, 224)):
        with torch.no_grad():  # Pas besoin de gradients ici
            sample_input = torch.randn(1, *input_shape)  # Crée un batch factice
            x = self.pool(F.relu(self.conv1(sample_input)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            return x.view(1, -1).shape[1]  # Retourne la taille aplatie


    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))  # Applique Dropout 
        x = self.fc2(x)
        return x
