import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops

folder_path = "C:/Users/ol1xy/2 course/2/comp_ml_vision/03.02/task_1"

sum_area = 0
for filename in os.listdir(folder_path):
    img_path = os.path.join(folder_path, filename)
    
    image = cv2.imread(img_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    '''
    label_image = label(gray_image)
    regions = regionprops(label_image)
    '''
    _, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    tmp_area = 0

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:

        area = cv2.contourArea(cnt)
        print(area)
        tmp_area+=area
    sum_area += tmp_area
      
print('sum area is ', sum_area)