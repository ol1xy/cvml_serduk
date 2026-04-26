from ultralytics import YOLO
import cv2
import time
from ultralytics.utils.plotting import Annotator
from playsound3 import playsound
import numpy as np

def get_angle(a, b, c):
    cb = np.atan2(c[1] - b[1], c[0] - b[0])
    ab = np.atan2(a[1] - b[1], a[0] - b[0])
    angle = np.rad2deg(cb - ab)
    angle = angle + 360 if angle < 0 else angle
    return 360 - angle if angle > 180 else angle

stage = None
threshhold_up = 160
threshhold_down = 115
counter = 0

def count_pushups(annotated_obj, keypoints):
    global counter
    global stage
    angles = []

    nose_seen = (keypoints[0][0] > 0 and keypoints[0][1] > 0)
    eyes_seen = (keypoints[1][0] > 0 and keypoints[1][1] > 0 and keypoints[2][0] > 0 and keypoints[2][1] > 0)
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]

    if (left_shoulder[1] > 0 and
            left_elbow[1] > 0) or (right_shoulder[1] > 0 and
            right_elbow[1] > 0):

            angle = get_angle(left_shoulder,
                                   left_elbow,
                                   left_wrist)
            angles.append(angle)
            if len(angles > 5):
                angles.pop(0)
            smooth_angle = np.mean(angles)
            
            if smooth_angle >= threshhold_up and stage == 'DOWN':
                stage = 'UP'
                counter += 1

            if smooth_angle <= threshhold_down:
                stage = 'DOWN'

                
    
    else:
        counter = 0
    cv2.putText(annotated_obj, f'{counter} pushups', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 1)

    return None

model = YOLO("yolo26n-pose.pt")
camera = cv2.VideoCapture(0)

ps = None

frame_count = 0
while camera.isOpened():
    ret, frame = camera.read()
    cv2.imshow("Camera", frame)
    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break

    t = time.perf_counter()

    if frame_count % 2 == 0:
        results = model(frame)
        if not results:
            continue

    result = results[0]
    keypoints = result.keypoints.xy.tolist()
    
    if not keypoints:
        continue

    frame_count += 1

    annotator = Annotator(frame)
    annotator.kpts(result.keypoints.data[0], result.orig_shape, 5, True)
    annotated = annotator.result()
    
    count_pushups(annotated, keypoints[0])

    cv2.imshow("Pose", annotated)
