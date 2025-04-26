import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms
from data_loader_detection import test_dataset
from polygon_detection_model import PolygonDetector

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Testing on: {device}")

model = PolygonDetector(num_points=4).to(device)
model.load_state_dict(torch.load("models/Detection/polygon_detector_ReduceLROnPlateau_2.pth", map_location=device))
model.eval()

def show_result(image, label, pred_bbox):
    image = image.permute(1, 2, 0).numpy()
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    width, height = image.shape[1], image.shape[0]
    true_coords = label[1:].cpu().numpy().reshape(-1, 2) * [width, height]
    pred_coords = pred_bbox.reshape(-1, 2) * [width, height]

    true_poly = patches.Polygon(true_coords, edgecolor='r', facecolor='none', linewidth=2)
    pred_poly = patches.Polygon(pred_coords, edgecolor='g', facecolor='none', linewidth=2)
    ax.add_patch(true_poly)
    ax.add_patch(pred_poly)

    ax.axis("off")
    plt.show()

for i in range(15):
    image, label = test_dataset[i]
    img_input = image.unsqueeze(0).to(device)

    with torch.no_grad():
        pred_bbox = model(img_input).squeeze().cpu().numpy()

    show_result(image, label, pred_bbox)