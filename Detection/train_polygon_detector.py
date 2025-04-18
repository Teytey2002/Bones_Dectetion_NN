import torch
import torch.nn as nn
import torch.optim as optim
import os
from data_loader_detection import train_loader, valid_loader
from polygon_detection_model import PolygonDetector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

model = PolygonDetector(num_points=4).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 20
os.makedirs("models", exist_ok=True)

for epoch in range(num_epochs):
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
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Valid Loss: {valid_epoch_loss:.4f}")

torch.save(model.state_dict(), "models/polygon_detector.pth")
print("Model saved to models/polygon_detector.pth")