from pathlib import Path

import cv2
import torch
from PIL import Image

from papzi.constants import MODEL_PATH
from papzi.models import ResNet50
from papzi.transforms import normalizing_transforms
from papzi.utils import load_label_map, torch_device


class FaceRecognizer:
    def __init__(self) -> None:
        from papzi.preprocess.normalizer import face_normalizer

        self.transforms = normalizing_transforms
        self.label_map = load_label_map()
        self.device = torch_device()
        self._load_model()
        self.face_normalizer = face_normalizer

    def _load_model(self):
        self.model = ResNet50(num_classes=len(self.label_map)).to(self.device)
        checkpoint = torch.load(MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, image_path: Path):

        image = self.face_normalizer.process_image(image_path)

        transform = normalizing_transforms

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # type: ignore
        image = Image.fromarray(image)
        image = transform(image)
        image = image.to(self.device)

        with torch.no_grad():
            outputs = self.model(image.unsqueeze(0))  # type: ignore
            _, predicted = torch.max(outputs, 1)
            print(outputs)
            class_idx = predicted.item()
            print(class_idx)
            class_name = self.label_map[str(class_idx)]
            print(class_name)
        return class_name
