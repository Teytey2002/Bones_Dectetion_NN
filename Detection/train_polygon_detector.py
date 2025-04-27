import torch
import torch.nn as nn
import torch.optim as optim
import os
from data_loader_detection import train_loader, valid_loader
from polygon_detection_model import PolygonDetector
from torch.utils.tensorboard import SummaryWriter

# Configuration 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")
model = PolygonDetector(num_points=4).to(device)
criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.9, verbose=True, threshold=1e-4)
writer = SummaryWriter(log_dir="runs_detection/polygon_detector_ReduceLROnPlateau_640ImageSize_dataAugm")    # Tensorboard

patience = 20
best_val_loss = float("inf")
epochs_no_improve = 0
epoch = 0
model_path = "models/Detection/polygon_detector_ReduceLROnPlateau_640ImageSize_dataAugm.pth"

# Train with Early stop
while True:
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        coords = labels[:, 1:]

        preds = model(images)
        loss = criterion(preds, coords)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            coords = labels[:, 1:]
            preds = model(images)
            loss = criterion(preds, coords)
            valid_loss += loss.item()

    valid_epoch_loss = valid_loss / len(valid_loader)
    scheduler.step(valid_epoch_loss)

    # Record curves on tensorboard
    writer.add_scalar("Loss/train", epoch_loss, epoch)
    writer.add_scalar("Loss/valid", valid_epoch_loss, epoch)
    writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], epoch)
    print(f"Epoch [{epoch+1}], Train Loss: {epoch_loss:.4f}, Valid Loss: {valid_epoch_loss:.4f}")
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