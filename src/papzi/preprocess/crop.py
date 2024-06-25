from pathlib import Path
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import random

# Define URLs for downloading the YuNet model
# YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2022mar.onnx"


# def download_model(url, MODEL_PATH):
#     if not MODEL_PATH.exists():
#         print(f"Downloading {MODEL_PATH.name}...")
#         urllib.request.urlretrieve(url, str(MODEL_PATH))
#         print(f"Downloaded {MODEL_PATH.name}")


# Paths to save the models
base_path = Path(__file__).absolute().parent
models_path = base_path / "models"
models_path.mkdir(parents=True, exist_ok=True)

yunet_MODEL_PATH = models_path / "face_detection_yunet_2023mar.onnx"

# Download the YuNet model if it does not exist
# download_model(YUNET_URL, yunet_MODEL_PATH)

# Load the YuNet model from OpenCV's model zoo
face_detector = cv2.FaceDetectorYN.create(
    model=str(yunet_MODEL_PATH),
    config="",
    input_size=(320, 320),
    score_threshold=0.9,
    nms_threshold=0.3,
    top_k=5000,
)

# Load the LBF face landmark detection model

facemark = cv2.face.createFacemarkLBF()
facemark.loadModel(cv2.samples.findFile("libfmodel.yaml"))


def crop_and_resize_face(image, bbox, target_size=(224, 224)):
    face = normalize_face_orientation(image, bbox)
    resized_face = cv2.resize(face, target_size)
    return resized_face


def normalize_face_orientation(image, bbox):
    (x, y, w, h) = bbox.astype(int)
    face = image[y : y + h, x : x + w]
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    _, landmarks = facemark.fit(gray, [bbox])

    if landmarks:
        landmarks = landmarks[0][0]
        left_eye = landmarks[36]
        right_eye = landmarks[45]

        eye_center_1 = (left_eye[0], left_eye[1])
        eye_center_2 = (right_eye[0], right_eye[1])

        if eye_center_1[0] > eye_center_2[0]:
            eye_center_1, eye_center_2 = eye_center_2, eye_center_1

        dy = eye_center_2[1] - eye_center_1[1]
        dx = eye_center_2[0] - eye_center_1[0]
        angle = cv2.fastAtan2(dy, dx)

        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
        face = cv2.warpAffine(face, M, (w, h))

    return face


def crop_per_class(cropped_class_dir: Path, class_dir: Path, num_images: int):
    image_files = list(class_dir.glob("*"))
    image_files = image_files[:num_images]  # Limit to the first `num_images`

    for image_path in image_files:
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        height, width, _ = image.shape
        face_detector.setInputSize((width, height))
        _, faces = face_detector.detect(image)

        if faces is not None and len(faces) > 0:
            # Use the first detected face
            face = crop_and_resize_face(image, faces[0][:4])
        else:
            # Resize the whole image if no face is detected
            face = cv2.resize(image, (224, 224))

        cropped_face_path = cropped_class_dir / image_path.name
        cv2.imwrite(str(cropped_face_path), face)


FROM_EACH_DIR = 1.0


def crop_faces_and_save(root_dir, cropped_dir, num_images=2000):
    root_dir = Path(root_dir)
    cropped_dir = Path(cropped_dir)

    if not cropped_dir.exists():
        cropped_dir.mkdir(parents=True, exist_ok=True)

    failed_count = 0
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = []
        full_dirs = list(root_dir.iterdir())
        random.shuffle(full_dirs)
        for class_dir in tqdm(
            full_dirs[: int(len(full_dirs) * FROM_EACH_DIR)]
        ):
            if class_dir.is_dir():
                cropped_class_dir = cropped_dir / class_dir.name
                if not cropped_class_dir.exists():
                    cropped_class_dir.mkdir(parents=True, exist_ok=True)

                futures.append(
                    executor.submit(
                        crop_per_class,
                        cropped_class_dir,
                        class_dir,
                        num_images,
                    )
                )
        t = tqdm(total=len(futures))
        for future in as_completed(futures):
            future.result()  # To catch exceptions if any
            t.update(1)

    print("failed", failed_count)


base = Path(__file__).absolute().parent / "data"
root_dir = base / "train"  # Path to your dataset
cropped_dir = base / "cropped"  # Path to save cropped faces
if __name__ == "__main__":
    crop_faces_and_save(root_dir, cropped_dir, num_images=500)
