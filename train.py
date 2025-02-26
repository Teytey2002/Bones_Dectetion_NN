import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from data_loader import train_loader, valid_loader  # Import des datasets
from model import BoneFractureCNN  # On sépare aussi le modèle dans un fichier à part
from torch.utils.tensorboard import SummaryWriter


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

# Tensorboard
writer = SummaryWriter(log_dir="runs/bone_fracture_experiment")

# Entraînement du modèle
for epoch in range(num_epochs):
    model.train()  # Mode entraînement
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    
    # Phase de validation
    model.eval()
    valid_loss = 0.0
    valid_correct = 0
    valid_total = 0
    
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            valid_correct += (predicted == labels).sum().item()
            valid_total += labels.size(0)
    
    valid_epoch_loss = valid_loss / len(valid_loader)
    valid_epoch_accuracy = 100 * valid_correct / valid_total

    # Enregistrement des courbes dans TensorBoard
    writer.add_scalar("Loss/train", epoch_loss, epoch)
    writer.add_scalar("Accuracy/train", epoch_accuracy, epoch)
    writer.add_scalar("Loss/valid", valid_epoch_loss, epoch)
    writer.add_scalar("Accuracy/valid", valid_epoch_accuracy, epoch)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, "
          f"Valid Loss: {valid_epoch_loss:.4f}, Valid Accuracy: {valid_epoch_accuracy:.2f}%")

# Sauvegarde du modèle
model_path = "models/bone_fracture_cnn.pth"
os.makedirs("models", exist_ok=True)  # Crée le dossier models s'il n'existe pas
torch.save(model.state_dict(), model_path)
print(f"Modèle sauvegardé sous {model_path}")
writer.close()  # Fermer TensorBoard

