from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from papzi.preprocess.normalizer import face_normalizer
from tqdm import tqdm

from papzi.utils import get_image_files


class FacePreProcessor:
    def __init__(self, root_dir, cropped_dir, num_images=30, max_workers=4):
        self.root_dir = Path(root_dir)
        self.cropped_dir = Path(cropped_dir)
        self.num_images = num_images
        self.max_workers = max_workers

    def ensure_dir_exists(self, directory):
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)

    def process_image(self, image_path, cropped_class_dir):
        normalized = face_normalizer.process_image(image_path)
        if normalized is not None:
            face_normalizer.save(
                normalized, cropped_class_dir / image_path.name
            )

    def process_class_dir(self, class_dir):
        cropped_class_dir = self.cropped_dir / class_dir.name
        self.ensure_dir_exists(cropped_class_dir)
        image_files = get_image_files(class_dir)[: self.num_images]

        for image_path in image_files:
            self.process_image(image_path, cropped_class_dir)

        print("done", class_dir.name)

    def crop_faces_and_save(self):
        self.ensure_dir_exists(self.cropped_dir)
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for class_dir in tqdm(list(self.root_dir.iterdir())):
                if class_dir.is_dir():

                    futures.append(
                        executor.submit(self.process_class_dir, class_dir)
                    )
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"An error occurred: {e}")


if __name__ == "__main__":
    base = Path(__file__).absolute().parent.parent / "data"
    root_dir = base / "scraped-old"  # Path to your dataset
    cropped_dir = base / "normalized"  # Path to save cropped faces
    face_cropper = FacePreProcessor(
        root_dir, cropped_dir, num_images=500, max_workers=8
    )
    face_cropper.crop_faces_and_save()
