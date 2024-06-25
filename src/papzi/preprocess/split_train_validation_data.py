import random
import shutil
from pathlib import Path

from papzi.constants import BASE_DIR


def move_images_for_validation(
    root_dir: Path, validation_dir: Path, num_images: int
):

    if not validation_dir.exists():
        validation_dir.mkdir(parents=True, exist_ok=True)

    for class_dir in root_dir.iterdir():
        if class_dir.is_dir():
            validation_class_dir = validation_dir / class_dir.name
            if not validation_class_dir.exists():
                validation_class_dir.mkdir(parents=True, exist_ok=True)
            else:
                continue
            image_files = list(class_dir.glob("*"))

            random.shuffle(image_files)
            validation_images = image_files[:num_images]

            for image in validation_images:
                dst_path = validation_class_dir / image.name
                shutil.move(str(image), str(dst_path))
                # print(
                #     f"Moved {image.name} from {class_dir}",
                #     " to {validation_class_dir}",
                # )
            print(class_dir.name)


root_dir = BASE_DIR / "train"  # Path to your dataset
validation_dir = BASE_DIR / "validation"  # Path to your validation directory
move_images_for_validation(root_dir, validation_dir, num_images=30)
