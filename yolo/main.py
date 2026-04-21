import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

classes = {0: "cube", 1: "neither", 2: "sphere"}

model = YOLO("runs/detect/figure_model/weights/best.pt")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    result = model(frame, conf = 0.5, verbose = False)

    for r in result:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # color = (0, 255, 0) if label == 'cube' else (0, 0, 255)
            color = (255, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label}, {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 2)
            
    cv2.imshow("Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()