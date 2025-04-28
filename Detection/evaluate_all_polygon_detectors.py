import torch
import os
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
from polygon_detection_model import PolygonDetector
from data_loader_detection import test_loader

# Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = "models/Detection/loop"
model_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]

results = []

for model_file in model_files:
    model_path = os.path.join(model_dir, model_file)
    model = PolygonDetector(num_points=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_mse = []
    all_distances = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Evaluating {model_file}"):
            images, labels = images.to(device), labels.to(device)
            preds = model(images)

            for i in range(images.size(0)):
                true_coords = labels[i, 1:].cpu().numpy()
                pred_coords = preds[i].cpu().numpy()

                mse = mean_squared_error(true_coords, pred_coords)
                dist = np.mean(np.sqrt((np.array(true_coords) - np.array(pred_coords))**2))

                all_mse.append(mse)
                all_distances.append(dist)

    avg_mse = np.mean(all_mse)
    avg_dist = np.mean(all_distances)

    results.append((model_file, avg_mse, avg_dist))
    print(f"✅ {model_file} : MSE moyen = {avg_mse:.6f}, Distance moyenne = {avg_dist:.6f}")

# Résumé final
print("\n📊 Résultats résumés :")
for model_file, avg_mse, avg_dist in results:
    print(f" - {model_file} : MSE = {avg_mse:.6f}, Distance = {avg_dist:.6f}")
