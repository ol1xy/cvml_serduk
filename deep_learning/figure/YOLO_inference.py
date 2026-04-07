from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

classes = {0: "neither", 1: "sphere", 2: "sphere"}
image_path = "/figure-dataset/ds/images/val/sphere/..."
model = YOLO("/runs/detect/figures/yolo/weights/best.pt")

plt.subplot(111)
image = np.array(Image.open(image_path).convert("RGB"))
plt.imshow(image)

result = model.predict(source = image_path,
                       conf = 0.25,
                       iou = 0.45,
                       imgsz = 640)[0]

boxes = result.boxes.xyxy.cpu.numpy()
cls = result.boxes.cls.cpu().numpy()
scores = result.boxes.conf.cpu().numpy()

for box, label, score in zip(boxes, cls, scores):
    x1, y1, x2, y2 = box
    rect = patches.Rectangle(
        (x1, y1), x2-x1, y2-y1, linewidth  = 2
    )
    plt.gca().add_patch(rect)
    plt.gca().text(x1, y1-10, f"{score:.2f}", color = "white",
                   fontsize = 12)
plt.show()