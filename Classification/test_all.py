
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from model import BoneFractureCNN
from data_loader import test_loader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = "models"
model_files = [f for f in os.listdir(model_dir) if f.startswith("best_model_") and f.endswith(".pth")]

criterion = torch.nn.CrossEntropyLoss()

best_accuracy = 0.0
best_model_path = None
best_preds = []
best_labels = []

# Test tous les modèles
for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)
    model = BoneFractureCNN(num_classes=7).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct = 0
    total = 0
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs, 1)

            test_loss += loss.item()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * correct / total
    print(f"✅ {model_file} — Test Accuracy: {accuracy:.2f}%")

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_path = model_path
        best_preds = all_preds
        best_labels = all_labels

# Résultat final
print(f"\n🏆 Meilleur modèle : {os.path.basename(best_model_path)} — Accuracy: {best_accuracy:.2f}%")

# Matrice de confusion
class_names = ['elbow', 'fingers', 'forearm fx', 'humerus fx', 'humerus', 'shoulder fx', 'wrist']
cm = confusion_matrix(best_labels, best_preds, labels=list(range(len(class_names))))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(xticks_rotation=45, cmap='Blues')
plt.title(f"Confusion Matrix — Best Model ({best_accuracy:.2f}%)")
plt.tight_layout()
plt.show()
