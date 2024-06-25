from pathlib import Path

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from papzi.transforms import normalizing_transforms
from papzi.utils import get_image_files


# Define custom dataset
class ImageDataset(Dataset):
    def __init__(
        self, root_dir: Path, num_images=600, num_classes: int | None = None
    ):
        self.root_dir = root_dir
        self.num_classes = num_classes
        self.num_images = num_images
        self.transform = normalizing_transforms
        # Define additional augmentations if enabled
        self.augmentation_transform = transforms.Compose(
            [
                # Add more augmentation transforms here as needed
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                # transforms.RandomAdjustSharpness()
            ]
        )

        self.labels = []
        self.label_map = {}
        self.images = []
        self._load_file_paths()
        self.classes = len(self.label_map)

    def __len__(self):
        return len(self.images)

    def _load_file_paths(self):
        label_id = 0
        actor_dirs = list(self.root_dir.iterdir())
        if self.num_classes is None:
            self.num_classes = len(actor_dirs)

        for class_path in tqdm(
            actor_dirs[: self.num_classes],
            desc=f"loading {self.root_dir.name} data",
            colour="blue",
        ):

            if not class_path.is_dir():
                continue

            image_paths = get_image_files(class_path, shuffle=True)[
                : self.num_images
            ]

            # image_paths.sort(key=lambda p: p.name)

            for image_name in image_paths:
                image_path = class_path / image_name

                self.images.append(self._load_image(image_path))
                self.labels.append(label_id)

            if class_path.name not in self.label_map.values():
                self.label_map[label_id] = class_path.name
                label_id += 1

    def _load_image(self, img_path: Path):

        # Convert image to RGBA if it has transparency
        image = Image.open(img_path)
        if image.mode == "P" and "transparency" in image.info:
            image = image.convert("RGBA")
        # else:
        image = image.convert("RGB")

        image = self.augmentation_transform(image)
        if self.transform:
            image = self.transform(image)

        return image

    def __getitem__(self, idx):
        label = self.labels[idx]
        return self.images[idx], label
