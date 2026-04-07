from ultralytics import YOLO
from pathlib import Path
import yaml
import torch_directml

if torch_directml.is_available():
    device = torch_directml.device()
else:
    device = "cpu"

classes = {0: "cube", 1: "neither", 2: "sphere"}

root = Path("figure-dataset/ds")

config = {
    "path" : str(root.absolute()),
    "train" : str((root / "images" / "train").absolute()),
    "val" : str((root / "images" / "val").absolute()),
    "nc" : len(classes),
    "names": classes
}

with open(root / "dataset.yaml", "w") as f:
    yaml.dump(config, f, allow_unicode=True)

size = "m"
model = YOLO(f"yolo26{size}.pt")

result = model.train(
    data = str(root / "dataset.yaml"),
    imgsz = 640, 
    batch = 16,
    workers = 0,

    epochs = 10,
    patience = 5,
    optimizer = "Adam",
    lr0 = 0.01,
    warmup_epochs = 3,
    cos_lr = True,

    dropout = 0.2,

    hsv_h = 0.015,
    hsv_s = 0.7,
    hsv_v = 0.4,
    flipud = 0.0,
    fliplr = 0.5, 
    mosaic = 1.0,
    degrees = 5.0,
    scale = 0.5,
    translate = 0.1,

    conf = 0.001,
    iou = 0.7,

    project = "figures",
    name = "yolo",
    save = True,
    save_period = 5,
    device = device,

    verbose = True,
    plots = True,
    val = True,
    close_mosaic = 8,
    amp = False #FP16
)

print(result.save_dir)
if __name__ == "__main__":
    train_process()
