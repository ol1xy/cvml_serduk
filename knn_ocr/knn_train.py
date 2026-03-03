import cv2
import numpy as np
from pathlib import Path
from skimage.measure import regionprops, label
from skimage.io import imread

CHAR_FIX = {
    "si": "i",
    "ss": "s",
    "sa": "a",
    "sc": "c",
    "sh": "h",
    "sk": "k",
    "sn": "n",
    "so": "o",
    "sp": "p",
    "sr": "r",
    "st": "t",
    "su": "u",
    "sv": "v",
    "sy": "y",
    "-": "-",
    "+": "+"
}

def extractor(image):
    if image.ndim == 3:
        image = np.mean(image, 2).astype("u1")
    binary = (image > 0).astype("u1")
    
    binary = cv2.dilate(binary, np.ones((3, 3), np.uint8))
    
    lb = label(binary)
    props = regionprops(lb)
    if not props: return None
    
    prop = max(props, key=lambda x: x.area)
    
    features = [
        prop.eccentricity, 
        prop.solidity, 
        prop.extent, 
        prop.orientation,
        prop.minor_axis_length / prop.major_axis_length if prop.major_axis_length > 0 else 0,
        prop.perimeter / prop.area if prop.area > 0 else 0
    ]
    return np.array(features, dtype="f4")

def train_model():

    base_path = Path(__file__).resolve().parent
    train_path = base_path / "task" / "train"
    
    train_data, responses, char_map = [], [], []
    subdirs = sorted([d for d in train_path.iterdir() if d.is_dir()])
    
    for idx, cls_dir in enumerate(subdirs):
        folder_name = cls_dir.name
        real_char = CHAR_FIX.get(folder_name, folder_name[0])
        char_map.append(real_char)
        
        for img_p in cls_dir.glob("*.png"):
            feat = extractor(imread(img_p))
            if feat is not None:
                train_data.append(feat)
                responses.append(idx)

    np.savez(base_path / "knn_data.npz", 
             train=np.array(train_data, dtype="f4"), 
             responses=np.array(responses, dtype="f4"),
             char_map=np.array(char_map))
    print(f"количество символов: {len(char_map)}")

train_model()