import json
import random
from pathlib import Path

import torch

from papzi.constants import BASE_DIR


def torch_device():
    # default device is cpu

    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available:
        return torch.device("cuda")
    return torch.device("cpu")


def load_label_map() -> dict[str, str]:
    with open(BASE_DIR / "classes.json") as f:
        return json.load(f)


def save_label_map(label_map: dict[str, str]):
    with open(BASE_DIR / "classes.json", "w") as f:
        json.dump(label_map, f)


def get_image_files(class_dir: Path, shuffle=False):
    types = ("*.jpg", "*.png", "*.jpeg")
    image_files = []
    for f_type in types:
        image_files.extend(class_dir.glob(f_type))
    if shuffle:

        random.shuffle(image_files)
    return image_files
