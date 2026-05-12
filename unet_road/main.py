import torch
import torch.nn as nn
import cv2
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from unet_road import UNet, RoadsDataset
from pathlib import Path


device = torch.device("cpu")
model = UNet()
model.load_state_dict(torch.load('unet_roads.pth', map_location=device))
model.eval()

path = Path("./roads/roads")
ds = RoadsDataset(path)


def show_prediction(idx):
    img, true_mask = ds[idx]

    with torch.no_grad():
        pred = model(img.unsqueeze(0).to(device))
        pred_mask = (torch.sigmoid(pred) > 0.5).float().squeeze().cpu().numpy()

    true_mask = true_mask.squeeze().numpy()
    delta = pred_mask - true_mask

    fig, ax = plt.subplots(1, 4, figsize = (20, 5))

    ax[0].imshow(img.permute(1, 2, 0).numpy())
    ax[0].set_title("orig")
    
    ax[1].imshow(true_mask, cmap = 'gray')
    ax[1].set_title("gt")

    ax[2].imshow(pred_mask, cmap = 'gray')
    ax[2].set_title("predicted")

    ax[3].imshow(delta, cmap = 'bwr', vmin = -1, vmax = 1)
    ax[3].set_title("delta (blue-miss, red-extra)")

    plt.show()

for i in range(3):
    show_prediction(i)

