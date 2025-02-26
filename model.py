import torch
import torch.nn as nn
import torch.nn.functional as F

class BoneFractureCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(BoneFractureCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # Fully Connected
        self.fc2 = nn.Linear(128, num_classes)  # Sortie
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
