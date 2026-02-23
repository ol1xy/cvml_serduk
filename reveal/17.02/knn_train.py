import cv2 
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.measure import regionprops, label
from skimage.io import imread
from collections import Counter

def extractor(image):

    if image.ndim == 2:
        binary = image

    else:
        gray = np.mean(image, 2).astype("u1")
        binary = gray < 255
    lb = label(binary)
    props = regionprops(lb)[0]

    radius = (props.area / np.pi) ** 0.5
    


    return np.array([props.eccentricity,
                     radius], dtype = 'f4')

def make_train(path):
    train = []
    responses = []
    ncls = 0

    for cls in sorted(path.glob("*")):
        ncls += 1
        print(cls.name, ncls)
        for img in cls.glob("*.png"):
            train.append(extractor(imread(img)))
            responses.append(ncls)
    train = np.array(train, dtype="f4").reshape(-1, 2)
    responses = np.array(responses, dtype="f4").reshape(-1, 1)
    return train, responses
            
# Добавить филлинг-фактор



data = Path("C:/Users/ol1xy/2_course/2/cvml_serduk/17.02/knn-example/out/train")

image = imread(data / "../image.png")

train, responses = make_train(data)

knn = cv2.ml.KNearest.create()
knn.train(train, cv2.ml.ROW_SAMPLE, responses)

gray = image.mean(2)
binary = gray < 255
lb = label(binary)
props = regionprops(lb)
find = []

for prop in props:
    find.append(extractor(prop.image))

find = np.array(find, dtype = "f4").reshape(-1, 2)

ret, results, neighbours, dist = knn.findNearest(
                                                find, 5)

print(Counter(np.array(results).flatten()))

plt.imshow(image)
plt.show()

