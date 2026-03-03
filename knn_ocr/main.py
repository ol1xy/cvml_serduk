import cv2
import numpy as np
from pathlib import Path
from skimage.measure import regionprops, label
from skimage.io import imread

base_path = Path(__file__).resolve().parent
data = np.load(base_path / "knn_data.npz")
knn = cv2.ml.KNearest_create()
knn.train(data['train'], cv2.ml.ROW_SAMPLE, data['responses'])
char_map = data['char_map']

def extractor(image_roi):
    lb = label(image_roi)
    props = regionprops(lb)
    if not props: return None
    prop = max(props, key=lambda x: x.area)
    return [prop.eccentricity, prop.solidity, prop.extent, prop.orientation,
            prop.minor_axis_length / prop.major_axis_length if prop.major_axis_length > 0 else 0,
            prop.perimeter / prop.area if prop.area > 0 else 0]

def process_images():
    task_path = base_path / "task"
    images = sorted([p for p in task_path.glob("*.png") if "train" not in p.parts])

    for img_p in images:
        image = imread(img_p)
        gray = np.mean(image, 2).astype("u1") if image.ndim == 3 else image
        
        binary = (gray > 0).astype("u1")
        
        binary = cv2.dilate(binary, np.ones((3, 3), np.uint8))
        
        lb = label(binary)
        props = sorted(regionprops(lb), key=lambda x: x.bbox[1])
        
        text, last_x_end = "", -1

        for prop in props:

            if prop.area < 25: continue
            if last_x_end != -1 and (prop.bbox[1] - last_x_end) > 20:
                text += " "

            feat = extractor(prop.image)
            if feat is not None:
                _, res, _, _ = knn.findNearest(np.array([feat], "f4"), 3)
                text += char_map[int(res[0][0])]
            
            last_x_end = prop.bbox[3]

        print(f"{img_p.name}: {text}")

process_images()