import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from datetime import datetime
from polygon_detection_model import PolygonDetector
from data_loader_detection import train_loader, valid_loader
from torch.utils.tensorboard import SummaryWriter

# Config
patience = 20
lr = 0.001
model_dir = "models/Detection/loop"
os.makedirs(model_dir, exist_ok=True)

# Fonction d'entraînement d'une session
def train_one_session():
    session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"runs_detection/loop/run_{session_id}"
    model_path = os.path.join(model_dir, f"polygon_detector_best_{session_id}.pth")

    writer = SummaryWriter(log_dir=log_dir)
    model = PolygonDetector(num_points=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.9, verbose=True, threshold=1e-4)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    epoch = 0

    print(f"🚀 Nouvelle session d'entraînement (détection) : {session_id}")
    try:
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

            # Logs TensorBoard
            writer.add_scalar("Loss/train", epoch_loss, epoch)
            writer.add_scalar("Loss/valid", valid_epoch_loss, epoch)

            print(f"Epoch [{epoch+1}], Train Loss: {epoch_loss:.4f}, Valid Loss: {valid_epoch_loss:.4f}")

            # EarlyStopping
            if valid_epoch_loss < best_val_loss:
                best_val_loss = valid_epoch_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), model_path)
                print(f">>> Nouveau meilleur modèle sauvegardé à l'epoch {epoch+1} (valid_loss: {best_val_loss:.4f})")
            else:
                epochs_no_improve += 1
                print(f"(Pas d'amélioration depuis {epochs_no_improve} epoch(s))")

            if epochs_no_improve >= patience:
                print(f"🛑 Early stopping déclenché après {patience} epochs sans amélioration.")
                break

            epoch += 1
            time.sleep(0.1)

    finally:
        writer.close()
        print(f"✅ Fin de session : {session_id} — modèle sauvegardé dans {model_path}\n")

# Entraînement en boucle continue
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        while True:
            train_one_session()
            print("⏳ Pause de 5 secondes avant prochaine session...")
            time.sleep(5)

    except KeyboardInterrupt:
        print("🛑 Entraînement global interrompu par l'utilisateur.")
