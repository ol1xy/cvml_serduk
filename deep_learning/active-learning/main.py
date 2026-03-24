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

device = torch_directml.device()
#TODO EFICCIENT_NET B 0 in HOMEWORK, size in entry parameters

def build_model():
    weights = torchvision.models.AlexNet_Weights.IMAGENET1K_V1
    model = torchvision.models.alexnet(weights=weights)
    for param in model.features.parameters():
        param.requires_grad = False

    features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(features, 1)
    return model.to(device=device)

model = build_model()
criterion = nn.BCEWithLogitsLoss()

optimizier = torch.optim.Adam(
    filter(lambda p: p.requires_grad,
           model.parameters()),
    lr = 0.0001
)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std = [0.229, 0.224, 0.225])
])

def train():
    pass

def predict(frame):
    model.eval()
    tensor = transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        predicted = model(tensor).squeeze()
        prob = torch.sigmoid(predicted).item()

    label = "person" if prob > 0.5 else "no_person"
    return label, prob




cap = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_GUI_NORMAL)
while True:
    _, frame = cap.read()
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(1) & 0xFf

    if key == ord("q"):
        break
    elif  key == ord("1"): #person
        pass
    elif key == ord("2"): #no person
        pass
    elif key == ord("p"): #predict
        pass
    elif key == ord("s"): #save_model
        pass
