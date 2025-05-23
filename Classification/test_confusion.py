import torch
import torch.nn as nn
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from Bones_Dectetion_NN.Classification.data_loader import test_loader  # Import du dataset test
from model import BoneFractureCNN  # Charger la même architecture

# Vérifier si un GPU est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing on: {device}")

# Charger le modèle
model = BoneFractureCNN(num_classes=7).to(device)
model.load_state_dict(torch.load("models/bone_fracture_cnn_dropout_conv3.pth"))  # Charger le modèle sauvegardé
model.eval()  # Mode évaluation
print("Modèle chargé et prêt pour le test.")

# Phase de test
test_loss = 0.0
test_correct = 0
test_total = 0
criterion = nn.CrossEntropyLoss()
all_preds = []
all_labels = []


for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    test_loss += loss.item()
    _, predicted = torch.max(outputs, 1)
    test_correct += (predicted == labels).sum().item()
    test_total += labels.size(0)

    all_preds.extend(predicted.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())


test_epoch_loss = test_loss / len(test_loader)
test_epoch_accuracy = 100 * test_correct / test_total

print(f"Test Loss: {test_epoch_loss:.4f}, Test Accuracy: {test_epoch_accuracy:.2f}%")

# Affichage des prédictions
def imshow(img):
    img = img / 2 + 0.5
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
imshow(torchvision.utils.make_grid(images.cpu()[:3]))
print("Labels réels :", labels.cpu().numpy()[:3])
print("Prédictions :", predicted.cpu().numpy()[:3])

# Matrice de confusion
class_names = ['elbow', 'fingers', 'forearm fx', 'humerus fx', 'humerus', 'shoulder fx', 'wrist']
cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(xticks_rotation=45, cmap='Blues')

plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
