from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('yolov8s.pt')  

    model.train(
        data=r'C:\Users\theod\OneDrive\Documents\ULB\Ma1\Project_Bones\DataSet\BoneFractureYolo8\data.yaml',
        epochs=150,
        imgsz=800,
        batch=16,
        name='yolo_bones_v8_s',
        project='runs/detect_s',
        exist_ok=True,
        patience=20,
        optimizer='Adam',
        verbose=True,
        device=0,
    )
