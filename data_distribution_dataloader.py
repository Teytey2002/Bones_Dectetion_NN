
import torch
from collections import Counter
from Classification.data_loader import train_loader, valid_loader, test_loader
import os

#def get_class_distribution(loader, name):
#    counter = Counter()
#    for _, labels in loader:
#        counter.update(labels.tolist())
#
#    print(f"\n📊 Répartition des classes dans le set {name}:")
#    total = sum(counter.values())
#    for cls in sorted(counter.keys()):
#        print(f"- Classe {cls}: {counter[cls]} images ({(counter[cls] / total * 100):.2f}%)")
#    return counter
#
#if __name__ == "__main__":
#    get_class_distribution(train_loader, "Train")
#    get_class_distribution(valid_loader, "Validation")
#    get_class_distribution(test_loader, "Test")




def count_images_and_labels(image_dir, label_dir):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    total_images = len(image_files)
    with_labels = 0
    for img_file in image_files:
        label_file = os.path.join(label_dir, img_file.replace('.jpg', '.txt'))
        if os.path.exists(label_file) and os.path.getsize(label_file) > 0:
            with_labels += 1
    without_labels = total_images - with_labels
    return total_images, with_labels, without_labels

base_path = r'C:\Users\theod\OneDrive\Documents\ULB\Ma1\Project_Bones\DataSet\BoneFractureYolo8'

for split in ['train', 'valid', 'test']:
    img_dir = os.path.join(base_path, split, 'images')
    lbl_dir = os.path.join(base_path, split, 'labels')
    total, with_lbl, without_lbl = count_images_and_labels(img_dir, lbl_dir)
    print(f"{split.upper()} SET → Total: {total}, With label: {with_lbl}, Without label: {without_lbl}")
