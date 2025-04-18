
import torch
from collections import Counter
from Classification.data_loader import train_loader, valid_loader, test_loader

def get_class_distribution(loader, name):
    counter = Counter()
    for _, labels in loader:
        counter.update(labels.tolist())

    print(f"\n📊 Répartition des classes dans le set {name}:")
    total = sum(counter.values())
    for cls in sorted(counter.keys()):
        print(f"- Classe {cls}: {counter[cls]} images ({(counter[cls] / total * 100):.2f}%)")
    return counter

if __name__ == "__main__":
    get_class_distribution(train_loader, "Train")
    get_class_distribution(valid_loader, "Validation")
    get_class_distribution(test_loader, "Test")
