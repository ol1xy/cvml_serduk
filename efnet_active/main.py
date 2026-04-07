import torch
import cv2
import torch.nn as nn
import torchvision
import numpy as np
from torchvision import transforms
from PIL import Image
import time
from collections import deque
import torch_directml
from pathlib import Path
import torchvision.models as models
from train_model import predict


model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(features, 1)
model.load_state_dict(torch.load('model.pth'))

cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)
count_labeled = 0

while True:
    _, frame = cap.read()
    key = cv2.waitKey(1) & 0xFf
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if key == ord("q"):
        break

    t = time.perf_counter()
    label, confidence = predict(frame, model)
    color = (0, 255, 0) if label == "person" else (0, 0, 255)
    cv2.putText(frame, f"{label} ({confidence:.2f})", (50, 50), 
        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    cv2.imshow("Camera", frame)
