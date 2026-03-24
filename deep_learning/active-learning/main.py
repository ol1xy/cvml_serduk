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
