import cv2
import numpy as np
import os
import json
import random

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

capture = cv2.VideoCapture(0 + cv2.CAP_DSHOW)
capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
capture.set(cv2.CAP_PROP_EXPOSURE, -7)


def get_color(image):
    x, y, w, h = cv2.selectROI("Color selection", image)
    x, y, w, h = int(x), int(y), int(w), int(h)
    roi = image[y:y+h, x:x+w]
    color = (np.median(roi[:, :, 0]),
            np.median(roi[:, :, 1]),
            np.median(roi[:, :, 2]))
    cv2.destroyWindow("Color selection")
    return color

def get_ball(image, color):
    lower = (np.max([0, color[0] - 5]), color[1] * 0.8, color[2] * 0.8)
    upper = (color[0] + 5, 255, 255)
    mask = cv2.inRange(image, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        contour = max(contours, key = cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(contour)
        return True, (int(x), int(y), int(radius), mask)
    return False, (-1, -1, -1, np.array([]))

cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)
file_name = "settings.json"
if os.path.exists(file_name):
    base_colors = json.load(open(file_name, "r"))
else:
    base_colors = {}

game_started = False
guess_colors = []

mode = 4

while capture.isOpened():
    ret, frame = capture.read()
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    key = chr(cv2.waitKey(1) & 0xFF)

    canvas = np.zeros_like(frame)


    color = get_color(hsv)
    base_colors[key] = color
    if key == "q":
        break
    subsequence = {}
    for key in base_colors:
        retr, (x, y, radius, mask) = get_ball(hsv, base_colors[key])
        subsequence[key] = x
        if retr:
            cv2.imshow("Mask", mask)
            cv2.circle(frame, (x, y), radius, (255, 0, 255), 2)

    while True:
        ok, frame = capture.read()
        if not ok: break

        hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (7,7), 0), cv2.COLOR_BGR2HSV)
        found, (x,y,r, _) = get_ball(hsv, color)

        if not found:
            prev = None
        else:
            cv2.circle(frame, (x,y), r, (255,0,255), 2)
            cv2.circle(frame, (x,y), 3, (0,255,0), -1)
            if prev:
                cv2.line(canvas, prev, (x,y), 255, int(r/10))
            prev = (x,y)

        k = cv2.waitKey(1)
        if k == ord('q'): break
        if k == ord('c'): canvas[:] = 0
        if k == ord('r'): color = get_color(hsv)

        cv2.imshow("Camera", frame)
        cv2.imshow("Canvas", canvas)

        capture.release(); cv2.destroyAllWindows()

        

    cv2.imshow("Camera", frame)

capture.release()
cv2.destroyAllWindows()

json.dump(base_colors, open(file_name, "w"))