from ultralytics import YOLO

def train():
    model = YOLO("yolo11n.pt")
    
    results = model.train(
        data = "dataset.yaml",
        workers = 0,
        epochs = 100,
        imgsz = 320,
        lr0 = 0.0005,
        batch = 16,
        name = "figure_model",
        optimizer = "SGD",
        momentum = 0.9,
        weight_decay=0.0005,
        patience = 10,
        save = True
    )

if __name__ == "__main__":
    train() 