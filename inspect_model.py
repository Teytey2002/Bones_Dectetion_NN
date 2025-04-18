import torch

# Chemin du modèle
model_path = "models/bone_fracture_cnn_dropout_conv3.pth"

# Chargement du state_dict
state_dict = torch.load(model_path, map_location="cpu")

print("\n--- Structure du modèle ---")
layer_shapes = {}
for k, v in state_dict.items():
    shape = tuple(v.shape)
    layer_shapes[k] = shape
    print(f"{k}: {shape}")

# Analyse simplifiée
print("\n--- Analyse rapide ---")
num_conv = sum(1 for name in state_dict if 'conv' in name and 'weight' in name)
num_bn = sum(1 for name in state_dict if 'bn' in name and 'weight' in name)
fc_shapes = [s for k, s in layer_shapes.items() if 'fc' in k and 'weight' in k]

print(f"Nombre de couches convolutionnelles: {num_conv}")
print(f"Présence de BatchNorm: {'oui' if num_bn > 0 else 'non'}")
if fc_shapes:
    print(f"Dimension d'entrée dans fc1: {fc_shapes[0][1]}")
    print(f"Dimension de sortie dans fc2: {fc_shapes[-1][0]}")

print("\n--- Conseils ---")
print("- Vérifie que le flatten_size correspond bien à l'entrée de fc1.")
print("- Vérifie que le modèle dans model.py a bien ces dimensions.")
