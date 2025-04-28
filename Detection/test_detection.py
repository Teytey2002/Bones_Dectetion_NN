import torch
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from data_loader_detection import test_loader
from polygon_detection_model import PolygonDetector


# Charger le modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PolygonDetector(num_points=4).to(device)

model.load_state_dict(torch.load("models/Detection/polygon_detector_ReduceLROnPlateau_640ImageSize_dataAugm.pth", map_location=device))
model.eval()

# Evaluation sur tout le test set
all_mse = []
all_distances = []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating"):
        images, labels = images.to(device), labels.to(device)

        # Prédictions
        preds = model(images)

        # Comparaison pour chaque image du batch
        for i in range(images.size(0)):
            true_coords = labels[i, 1:].cpu().numpy()
            pred_coords = preds[i].cpu().numpy()

            mse = mean_squared_error(true_coords, pred_coords)
            dist = np.mean(np.sqrt((np.array(true_coords) - np.array(pred_coords))**2))

            all_mse.append(mse)
            all_distances.append(dist)

# Résumé des métriques
avg_mse = np.mean(all_mse)
avg_dist = np.mean(all_distances)

print(f"✅ Evaluation terminée :")
print(f" - MSE moyen des polygones : {avg_mse:.6f}")
print(f" - Distance moyenne entre points : {avg_dist:.6f}")