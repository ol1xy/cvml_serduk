import shutil
from pathlib import Path
import subprocess

base = Path("spheres_and_cubes_new")

for split in ["train", "val"]:
    for data_type in ["images", "labels"]:
        parent = base / data_type / split
        for file in parent.rglob("*"):
            if file.is_file():
                shutil.move(str(file), str(parent / file.name))
        
        for sub in ["cube", "neither", "sphere"]:
            if (parent / sub).exists():
                shutil.rmtree(parent / sub)

subprocess.run(["tree",  str(base)], shell=True)