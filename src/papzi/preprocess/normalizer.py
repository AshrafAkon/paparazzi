from pathlib import Path

import cv2
import numpy as np


class FaceNormalizer:
    def __init__(
        self,
        face_detector_model: Path,
        facemark_model: Path,
        target_size=(224, 224),
    ):
        """
        Initialize the FaceNormalizer with face detector and facemark models.
        """

        # Download the YuNet model if it does not exist
        # download_model(YUNET_URL, yunet_MODEL_PATH)

        # Load the YuNet model from OpenCV's model zoo
        self.face_detector = cv2.FaceDetectorYN.create(
            model=str(face_detector_model),
            config="",
            input_size=(320, 320),
            score_threshold=0.9,
            nms_threshold=0.3,
            top_k=5000,
        )

        # Load the LBF face landmark detection model

        self.facemark = cv2.face.createFacemarkLBF()
        self.facemark.loadModel(str(facemark_model.resolve()))

        self.target_size = target_size

    def load_image(self, image_path):
        """
        Load and convert the image to RGB format.
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, image_rgb

    def detect_faces(self, image):
        """
        Detect faces in the image using the face detector.
        """
        height, width, _ = image.shape
        self.face_detector.setInputSize((width, height))
        faces = self.face_detector.detect(image)
        return faces

    def expand_bbox(self, bbox, image_shape, scale=1.0):
        """
        Expand the bounding box by a given scale factor.
        """
        x, y, w, h = bbox.astype(int)
        new_w = int(w * scale)
        new_h = int(h * scale)
        new_x = max(0, x - (new_w - w) // 2)
        new_y = max(0, y - (new_h - h) // 2)
        new_w = min(new_w, image_shape[1] - new_x)
        new_h = min(new_h, image_shape[0] - new_y)
        return np.array([new_x, new_y, new_w, new_h])

    def detect_landmarks(self, face, bbox):
        """
        Detect facial landmarks within the face region.
        """
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        success, landmarks = self.facemark.fit(gray, np.array([bbox]))
        if success and len(landmarks) > 0:
            return landmarks[0][0]
        else:
            raise ValueError("Failed to detect landmarks.")

    def get_rotation_matrix(self, eye_center_1, eye_center_2, center):
        """
        Compute the rotation matrix to align the eyes horizontally.
        """
        if eye_center_1[0] > eye_center_2[0]:
            eye_center_1, eye_center_2 = eye_center_2, eye_center_1
        dy = eye_center_2[1] - eye_center_1[1]
        dx = eye_center_2[0] - eye_center_1[0]
        angle = np.degrees(np.arctan2(dy, dx))
        return cv2.getRotationMatrix2D(center, angle, 1)

    def transform_bbox(self, bbox_expanded_points, M, image_shape):
        """
        Transform the bounding box coordinates using the rotation matrix.
        """
        transformed_bbox_points = cv2.transform(
            np.array([bbox_expanded_points]), M
        )[0]
        x_min, y_min = transformed_bbox_points.min(axis=0)
        x_max, y_max = transformed_bbox_points.max(axis=0)
        x_min, y_min = max(0, int(x_min)), max(0, int(y_min))
        x_max, y_max = min(image_shape[1], int(x_max)), min(
            image_shape[0], int(y_max)
        )
        return x_min, y_min, x_max, y_max

    def pad_face(self, face, warped_image, x_min, y_min, x_max, y_max):
        """
        Pad the cropped face image using the pixels from the warped image.
        """
        h, w, _ = face.shape
        target_h, target_w = self.target_size

        # Calculate padding sizes
        pad_top = (target_h - h) // 2
        pad_bottom = target_h - h - pad_top
        pad_left = (target_w - w) // 2
        pad_right = target_w - w - pad_left

        # Create a new image with the target size
        padded_face = np.zeros((target_h, target_w, 3), dtype=face.dtype)

        # Insert the original face into the center of the new image
        padded_face[pad_top : pad_top + h, pad_left : pad_left + w] = face

        # Pad the top
        if pad_top > 0:
            padded_face[:pad_top, pad_left : pad_left + w] = warped_image[
                max(0, y_min - pad_top) : y_min, x_min:x_max
            ]

        # Pad the bottom
        if pad_bottom > 0:
            padded_face[pad_top + h :, pad_left : pad_left + w] = warped_image[
                y_max : min(y_max + pad_bottom, warped_image.shape[0]),
                x_min:x_max,
            ]

        # Pad the left
        if pad_left > 0:
            padded_face[:, :pad_left] = warped_image[
                y_min:y_max, max(0, x_min - pad_left) : x_min
            ]

        # Pad the right
        if pad_right > 0:
            padded_face[:, pad_left + w :] = warped_image[
                y_min:y_max,
                x_max : min(x_max + pad_right, warped_image.shape[1]),
            ]

        return padded_face

    def resize_and_pad_face(self, face):
        """
        Resize the face while maintaining the
        aspect ratio and pad to the target size.
        """
        target_h, target_w = self.target_size
        h, w, _ = face.shape
        scale = min(target_h / h, target_w / w)
        resized_h, resized_w = int(h * scale), int(w * scale)

        # Resize the face
        resized_face = cv2.resize(face, (resized_w, resized_h))

        # Create a new image with the target size
        padded_face = np.zeros(
            (target_h, target_w, 3), dtype=resized_face.dtype
        )

        # Calculate padding sizes
        pad_top = (target_h - resized_h) // 2
        pad_bottom = target_h - resized_h - pad_top
        pad_left = (target_w - resized_w) // 2
        pad_right = target_w - resized_w - pad_left

        # Insert the resized face into the center of the new image
        padded_face[
            pad_top : pad_top + resized_h, pad_left : pad_left + resized_w
        ] = resized_face

        # Fill the padding regions with the surrounding
        # pixels from the resized face
        if pad_top > 0:
            padded_face[:pad_top, pad_left : pad_left + resized_w] = (
                resized_face[0:1, :, :]
            )
        if pad_bottom > 0:
            padded_face[
                pad_top + resized_h :, pad_left : pad_left + resized_w
            ] = resized_face[-1:, :, :]
        if pad_left > 0:
            padded_face[:, :pad_left] = padded_face[:, pad_left : pad_left + 1]
        if pad_right > 0:
            padded_face[:, pad_left + resized_w :] = padded_face[
                :, pad_left + resized_w - 1 : pad_left + resized_w
            ]

        return padded_face

    def make_square_bbox(self, bbox, image_shape):
        """
        Adjust the bounding box to make it square, centered
        on the original bounding box.
        """
        x, y, w, h = bbox.astype(int)
        if w > h:
            diff = w - h
            y = max(0, y - diff // 2)
            h = w
        else:
            diff = h - w
            x = max(0, x - diff // 2)
            w = h
        # Ensure the bbox is within the image boundaries
        x = min(x, image_shape[1] - w)
        y = min(y, image_shape[0] - h)
        return np.array([x, y, w, h])

    def normalize_face_orientation(self, image, bbox):
        """
        Normalize the orientation of the face by
        aligning the eyes horizontally.
        """
        bbox = self.make_square_bbox(bbox, image.shape)
        bbox_expanded = self.expand_bbox(bbox, image.shape)

        x, y, w, h = bbox_expanded.astype(int)

        # Extract the face region
        face = image[y : y + h, x : x + w]

        # Detect landmarks
        landmarks = self.detect_landmarks(face, [0, 0, w, h])
        left_eye, right_eye = landmarks[36], landmarks[45]
        eye_center_1 = (int(left_eye[0]), int(left_eye[1]))
        eye_center_2 = (int(right_eye[0]), int(right_eye[1]))

        # Compute rotation matrix
        center = (image.shape[1] // 2, image.shape[0] // 2)
        M = self.get_rotation_matrix(eye_center_1, eye_center_2, center)

        # Warp the whole image
        warped_image = cv2.warpAffine(
            image, M, (image.shape[1], image.shape[0])
        )

        # Transform the bounding box
        bbox_expanded_points = np.array(
            [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]
        )
        x_min, y_min, x_max, y_max = self.transform_bbox(
            bbox_expanded_points, M, image.shape
        )

        # Crop the face from the warped image
        face_warped = warped_image[y_min:y_max, x_min:x_max]

        # Resize and pad the cropped face to the target size
        normalized_face = self.resize_and_pad_face(face_warped)
        return normalized_face

    def process_image(self, image_path):
        """
        Process an image to detect and normalize faces.
        """
        image, image_rgb = self.load_image(image_path)
        faces = self.detect_faces(image)
        if faces[1] is not None:
            for face in faces[1]:
                bbox = face[:4].astype(int)

                bbox_expanded = self.expand_bbox(bbox, image.shape)
                cv2.rectangle(
                    image_rgb,
                    (bbox_expanded[0], bbox_expanded[1]),
                    (
                        bbox_expanded[0] + bbox_expanded[2],
                        bbox_expanded[1] + bbox_expanded[3],
                    ),
                    (255, 0, 0),
                    2,
                )

                try:
                    return self.normalize_face_orientation(image, bbox)

                    # plt.imshow(
                    #     cv2.cvtColor(normalized_face, cv2.COLOR_BGR2RGB)
                    # )
                    # plt.axis("off")
                    # plt.show()
                except Exception as e:
                    raise e
                    print(f"Error normalizing face: {e}")

        # plt.imshow(image_rgb)
        # plt.axis("off")
        # plt.show()

    def save(self, normalized_face, output_path: Path):
        """
        Save the normalized face image to the specified output path.
        """
        cv2.imwrite(
            str(output_path.resolve()),
            normalized_face,
        )


# Example usage:
# face_normalizer = FaceNormalizer("yunet.onnx", "lbfmodel.yaml")
# face_normalizer.process_image("path_to_your_image.jpg")

models_path = Path(__file__).parent / "models"
print(
    models_path / "face_detection_yunet_2023mar.onnx",
    models_path / "lbfmodel.yaml",
)
# Example usage:
face_normalizer = FaceNormalizer(
    models_path / "face_detection_yunet_2023mar.onnx",
    models_path / "lbfmodel.yaml",
)
