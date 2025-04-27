import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
from datetime import datetime
from model import BoneFractureCNN
from data_loader import train_loader, valid_loader
from torch.utils.tensorboard import SummaryWriter

# Config
patience = 20
lr = 0.001
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Fonction d'entraînement d'une session
def train_one_session():
    # Générer un nom horodaté
    session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"runs/run_{session_id}"
    model_path = os.path.join(model_dir, f"best_model_{session_id}.pth")

    # Initialisation
    writer = SummaryWriter(log_dir=log_dir)
    model = BoneFractureCNN(num_classes=7).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    epochs_no_improve = 0
    epoch = 0

    print(f"🚀 Nouvelle session d'entraînement : {session_id}")
    try:
        while True:
            model.train()
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
            epoch_acc = 100 * correct / total

            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in valid_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)

            val_epoch_loss = val_loss / len(valid_loader)
            val_epoch_acc = 100 * val_correct / val_total

            # Logs TensorBoard
            writer.add_scalar("Loss/train", epoch_loss, epoch)
            writer.add_scalar("Accuracy/train", epoch_acc, epoch)
            writer.add_scalar("Loss/valid", val_epoch_loss, epoch)
            writer.add_scalar("Accuracy/valid", val_epoch_acc, epoch)

            print(f"Epoch [{epoch+1}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%, "
                  f"Valid Loss: {val_epoch_loss:.4f}, Valid Accuracy: {val_epoch_acc:.2f}%")

            # EarlyStopping
            if val_epoch_acc > best_val_acc:
                best_val_acc = val_epoch_acc
                epochs_no_improve = 0
                torch.save(model.state_dict(), model_path)
                print(f">>> Nouveau meilleur modèle sauvegardé à l'epoch {epoch+1} (acc: {best_val_acc:.2f}%)")
            else:
                epochs_no_improve += 1
                print(f"(Pas d'amélioration depuis {epochs_no_improve} epoch)")

            if epochs_no_improve >= patience:
                print(f"🛑 Early stopping déclenché après {patience} epochs sans amélioration.")
                break

            epoch += 1
            time.sleep(0.1)

    finally:
        writer.close()
        print(f"✅ Fin de session : {session_id} — modèle dans {model_path}\n")

# Entraînement en boucle continue
if __name__ == "__main__":
    #torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        while True:
            train_one_session()
            print("⏳ Pause de 5 secondes avant prochaine session...")
            time.sleep(5)

    except KeyboardInterrupt:
        print("🛑 Entraînement global interrompu par l'utilisateur. À demain champion.ne !")
