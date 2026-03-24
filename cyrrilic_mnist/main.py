import torch
import torch.nn as nn
import cv2
import os
import numpy as np
from train_model import CyrillicCNN
from torchvision import transforms

dataset_path = "cyrillic/Cyrillic"
classes = sorted(os.listdir(dataset_path))

model = CyrillicCNN(len(classes))
model.load_state_dict(torch.load("cyrillic_model.pth",
map_location = torch.device('cpu')))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

canvas = np.zeros((256, 256), dtype = "uint8")
cv2.namedWindow("Canvas", cv2.WINDOW_GUI_NORMAL)

draw = False

def on_mouse(event, x, y, flags, param):
    global draw
    if event == cv2.EVENT_LBUTTONDOWN: draw = True
    if event == cv2.EVENT_LBUTTONUP: draw = False
    if event == cv2.EVENT_MOUSEMOVE and draw:
        cv2.circle(canvas, (x, y), 8, 255, -1)

cv2.setMouseCallback("Canvas", on_mouse)

while True:
    display_img = canvas.copy()

    tensor = transform(canvas)
    batch = tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(batch)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs).item()
        confidence = torch.max(probs).item()

        if confidence > 0.8:
            text = f"{classes[pred_idx]} ({int(confidence * 100)}%)"
            # cv2.putText(display_img, text, (10, 40), 
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255), 2)
            print(f"Буква: {classes[pred_idx]}")
    cv2.imshow("Canvas", display_img)

    key = cv2.waitKey(1) & 0xFF
    if key == 27: break
    if key == ord('c'):
        canvas[:] = 0
cv2.destroyAllWindows()