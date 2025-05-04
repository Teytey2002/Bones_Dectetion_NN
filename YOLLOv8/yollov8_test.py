from ultralytics import YOLO

if __name__ == "__main__":
    # Charger ton meilleur modèle
    model = YOLO('runs/detect_s/yolo_bones_v8_s/weights/best.pt')

    # Validation sur le set de test (pas sur valid)
    metrics = model.val(
        data=r'C:\Users\theod\OneDrive\Documents\ULB\Ma1\Project_Bones\DataSet\BoneFractureYolo8\data.yaml', 
        split='test'  # Important : utiliser le dossier test !
    )


