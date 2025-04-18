import torch
import torch.nn as nn
import torch.optim as optim
import os
from data_loader_detection import train_loader, valid_loader  # Import du DataLoader adapté
from model_detection import BoneFractureDetector  # Import du modèle de détection

# Vérifier si un GPU est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Initialisation du modèle
num_classes = 7
num_points = 4  # 4 points -> 8 coordonnées
model = BoneFractureDetector(num_classes=num_classes, num_points=num_points).to(device)

# Définition des fonctions de perte
criterion_class = nn.CrossEntropyLoss()  # Classification
criterion_bbox = nn.MSELoss()  # Prédiction des polygones

# Optimiseur
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Nombre d'epochs
num_epochs = 20

# Dossier pour sauvegarder le modèle
os.makedirs("models", exist_ok=True)

# Entraînement du modèle
for epoch in range(num_epochs):
    model.train()  # Mode entraînement
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Séparation des labels
        class_labels = labels[:, 0].long()  # Classes (entiers)
        bbox_labels = labels[:, 1:]  # Coordonnées du polygone (float)

        # Prédictions du modèle
        class_preds, bbox_preds = model(images)

        # Calcul des pertes
        loss_class = criterion_class(class_preds, class_labels)
        loss_bbox = criterion_bbox(bbox_preds, bbox_labels)
        loss = loss_class + loss_bbox  # Somme des pertes

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Précision (classification)
        _, predicted = torch.max(class_preds, 1)
        correct += (predicted == class_labels).sum().item()
        total += class_labels.size(0)

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

            class_labels = labels[:, 0].long()
            bbox_labels = labels[:, 1:]

            class_preds, bbox_preds = model(images)

            loss_class = criterion_class(class_preds, class_labels)
            loss_bbox = criterion_bbox(bbox_preds, bbox_labels)
            loss = loss_class + loss_bbox

            valid_loss += loss.item()

            _, predicted = torch.max(class_preds, 1)
            valid_correct += (predicted == class_labels).sum().item()
            valid_total += class_labels.size(0)

    valid_epoch_loss = valid_loss / len(valid_loader)
    valid_epoch_accuracy = 100 * valid_correct / valid_total

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%, "
          f"Valid Loss: {valid_epoch_loss:.4f}, Valid Accuracy: {valid_epoch_accuracy:.2f}%")

# Sauvegarde du modèle
model_path = "models/bone_fracture_detection.pth"
torch.save(model.state_dict(), model_path)
print(f"Modèle sauvegardé sous {model_path}")
