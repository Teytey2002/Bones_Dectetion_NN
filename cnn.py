import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from data_loader import train_loader, valid_loader, test_loader


class BoneFractureCNN(nn.Module):
    def __init__(self, num_classes=7):  # 7 classes
        super(BoneFractureCNN, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 56 * 56, 128)  # 64 canaux * 56x56 après MaxPooling
        self.fc2 = nn.Linear(128, num_classes)  # Dernière couche avec 7 classes
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # Bloc Conv1
        x = self.pool(F.relu(self.conv2(x)))  # Bloc Conv2
        x = torch.flatten(x, 1)  # Mise à plat des features
        x = F.relu(self.fc1(x))  # Fully Connected Layer 1
        x = self.fc2(x)  # Fully Connected Layer 2 (sortie finale)
        return x

# Vérifier si un GPU est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Initialisation du modèle
model = BoneFractureCNN(num_classes=7).to(device)

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Nombre d'epochs
num_epochs = 10

# Boucle d'entraînement
for epoch in range(num_epochs):
    model.train()  # Mode entraînement
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # Envoyer sur GPU si dispo
        
        # Étape 1: Forward pass
        outputs = model(images)
        
        # Étape 2: Calcul de la perte
        loss = criterion(outputs, labels)
        
        # Étape 3: Backpropagation et optimisation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistiques
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)  # Prendre la classe avec la plus grande probabilité
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    # Calcul des métriques d'entraînement
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    
    # Phase de validation
    model.eval()  # Mode évaluation
    valid_loss = 0.0
    valid_correct = 0
    valid_total = 0
    
    with torch.no_grad():  # Pas de calcul de gradient en validation
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            valid_correct += (predicted == labels).sum().item()
            valid_total += labels.size(0)
    
    # Calcul des métriques de validation
    valid_epoch_loss = valid_loss / len(valid_loader)
    valid_epoch_accuracy = 100 * valid_correct / valid_total
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, "
          f"Valid Loss: {valid_epoch_loss:.4f}, Valid Accuracy: {valid_epoch_accuracy:.2f}%")
    

    ## Affichage des résultats par epoch
    #epoch_loss = running_loss / len(train_loader)
    #epoch_accuracy = 100 * correct / total
    #print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

print("Entraînement terminé !")


# Mode évaluation
model.eval()

test_loss = 0.0
test_correct = 0
test_total = 0

with torch.no_grad():  # Pas de calcul de gradient en test
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)  # Calcul de la perte
        
        test_loss += loss.item()
        _, predicted = torch.max(outputs, 1)  # Classe avec la plus grande probabilité
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

# Calcul des métriques
test_epoch_loss = test_loss / len(test_loader)
test_epoch_accuracy = 100 * test_correct / test_total

print(f"Test Loss: {test_epoch_loss:.4f}, Test Accuracy: {test_epoch_accuracy:.2f}%")

# Fonction pour afficher une image
def imshow(img):
    img = img / 2 + 0.5  # Dé-normalisation
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    plt.show()

# Récupérer un batch d'images test
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Prédictions du modèle
images, labels = images.to(device), labels.to(device)
outputs = model(images)
_, predicted = torch.max(outputs, 1)

# Affichage
imshow(torchvision.utils.make_grid(images.cpu()[:8]))  # Afficher 8 images
print("Labels réels :", labels.cpu().numpy()[:8])
print("Prédictions :", predicted.cpu().numpy()[:8])
