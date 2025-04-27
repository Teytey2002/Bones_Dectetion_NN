import torch
import torch.nn as nn
import torch.optim as optim
import os
from data_loader_detection_classification import train_loader, valid_loader
from model_detection_classification import BoneFractureMultiTaskDeepNet
from torch.utils.tensorboard import SummaryWriter

# Configuration 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Initialisation du modèle
model = BoneFractureMultiTaskDeepNet(num_classes=7, num_points=4).to(device)
criterion_class = nn.CrossEntropyLoss()
criterion_bbox = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=10, verbose=True)

# TensorBoard
writer = SummaryWriter(log_dir="runs_multitask/bone_fracture_multitask_conv10_normalized_schedule_0.6dp")

# Early stopping
patience = 25
best_val_loss = float("inf")
epochs_no_improve = 0
epoch = 0
model_path = "models/Detection_Classification/bone_fracture_multitask_conv10_normalized_schedule_0.6dp.pth"
os.makedirs(os.path.dirname(model_path), exist_ok=True)

# Entraînement
while True:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        class_labels = labels[:, 0].long()
        bbox_labels = labels[:, 1:]

        class_preds, bbox_preds = model(images)

        loss_class = criterion_class(class_preds, class_labels)
        loss_bbox = criterion_bbox(bbox_preds, bbox_labels)
        loss = loss_class + loss_bbox

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(class_preds, 1)
        correct += (predicted == class_labels).sum().item()
        total += class_labels.size(0)

    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total

    # Validation
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
    scheduler.step(valid_epoch_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"📉 Learning Rate actuel: {current_lr:.6f}")


    # TensorBoard recording
    writer.add_scalar("Loss/train", epoch_loss, epoch)
    writer.add_scalar("Loss/valid", valid_epoch_loss, epoch)
    writer.add_scalar("Accuracy/train", epoch_accuracy, epoch)
    writer.add_scalar("Accuracy/valid", valid_epoch_accuracy, epoch)
    writer.add_scalar("LearningRate", current_lr, epoch)


    print(f"Epoch [{epoch+1}], Train Loss: {epoch_loss:.4f}, Valid Loss: {valid_epoch_loss:.4f}, \
          Train Acc: {epoch_accuracy:.2f}%, Valid Acc: {valid_epoch_accuracy:.2f}%")
    epoch += 1

    # Early stopping
    if valid_epoch_loss < best_val_loss:
        best_val_loss = valid_epoch_loss
        torch.save(model.state_dict(), model_path)
        print(f"✅ Nouveau meilleur modèle sauvegardé (loss={best_val_loss:.4f})")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"⏸️ Pas d'amélioration depuis {epochs_no_improve} epoch(s)")

    if epochs_no_improve >= patience:
        print("🛑 Early stopping déclenché")
        break

writer.close()
