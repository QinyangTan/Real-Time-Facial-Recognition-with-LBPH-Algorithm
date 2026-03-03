from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

# Default paths
_MODEL_PATH = "trainingData.yml"
_LOCAL_CASCADE = Path("HaarCascade") / "haarcascade_frontalface_default.xml"
_OPENCV_CASCADE = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"

# Reuse expensive objects instead of rebuilding them every frame.
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_FACE_CASCADE: Optional[cv2.CascadeClassifier] = None


Rect = Tuple[int, int, int, int]


def _resolve_cascade_path(custom_path: Optional[str] = None) -> str:
    if custom_path and Path(custom_path).exists():
        return custom_path
    if _LOCAL_CASCADE.exists():
        return str(_LOCAL_CASCADE)
    return str(_OPENCV_CASCADE)


def get_face_cascade(custom_path: Optional[str] = None) -> cv2.CascadeClassifier:
    global _FACE_CASCADE

    if custom_path is not None:
        cascade = cv2.CascadeClassifier(_resolve_cascade_path(custom_path))
        if cascade.empty():
            raise FileNotFoundError("Unable to load Haar cascade classifier.")
        return cascade

    if _FACE_CASCADE is None:
        _FACE_CASCADE = cv2.CascadeClassifier(_resolve_cascade_path())
        if _FACE_CASCADE.empty():
            raise FileNotFoundError("Unable to load Haar cascade classifier.")

    return _FACE_CASCADE


def preprocess_frame(image: np.ndarray) -> np.ndarray:
    """Convert an image to grayscale and apply CLAHE for better lighting robustness."""
    if image is None:
        raise ValueError("Input image is None.")

    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return _CLAHE.apply(gray)


def extract_face_roi(gray_img: np.ndarray, face: Rect, target_size: Optional[Tuple[int, int]] = (200, 200)) -> np.ndarray:
    x, y, w, h = face
    roi = gray_img[y : y + h, x : x + w]

    if roi.size == 0:
        raise ValueError("Empty face ROI extracted.")

    if target_size is not None:
        roi = cv2.resize(roi, target_size, interpolation=cv2.INTER_AREA)

    return roi


def faceDetection(
    test_img: np.ndarray,
    scaleFactor: float = 1.2,
    minNeighbors: int = 5,
    minSize: Tuple[int, int] = (60, 60),
    cascade_path: Optional[str] = None,
) -> Tuple[List[Rect], np.ndarray]:
    """Backward-compatible helper used by the original scripts."""
    gray_img = preprocess_frame(test_img)
    face_haar_cascade = get_face_cascade(cascade_path)
    faces = face_haar_cascade.detectMultiScale(
        gray_img,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize,
    )
    return [tuple(map(int, face)) for face in faces], gray_img


def labels_for_training_data(
    directory: str,
    image_size: Tuple[int, int] = (200, 200),
) -> Tuple[List[np.ndarray], List[int]]:
    """Load training images from label folders and return face ROIs plus integer labels."""
    faces: List[np.ndarray] = []
    face_ids: List[int] = []

    for path, _, filenames in os.walk(directory):
        filenames = sorted(filenames)
        label_name = os.path.basename(path)

        # Skip the root folder and any non-numeric folder names.
        if path == directory or not label_name.isdigit():
            continue

        label = int(label_name)

        for filename in filenames:
            if filename.startswith("."):
                continue

            img_path = os.path.join(path, filename)
            image = cv2.imread(img_path)
            if image is None:
                continue

            faces_rect, gray_img = faceDetection(image)
            if not faces_rect:
                continue

            # If multiple faces are found, use the largest one.
            face = max(faces_rect, key=lambda rect: rect[2] * rect[3])

            try:
                roi_gray = extract_face_roi(gray_img, face, target_size=image_size)
            except ValueError:
                continue

            faces.append(roi_gray)
            face_ids.append(label)

    return faces, face_ids


def train_classifier(
    faces: Sequence[np.ndarray],
    faceID: Sequence[int],
    radius: int = 1,
    neighbors: int = 8,
    grid_x: int = 8,
    grid_y: int = 8,
):
    if not faces:
        raise ValueError("No training faces found. Check your trainingImages directory.")

    face_recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=radius,
        neighbors=neighbors,
        grid_x=grid_x,
        grid_y=grid_y,
    )
    face_recognizer.train(list(faces), np.array(faceID, dtype=np.int32))
    return face_recognizer


def save_model(face_recognizer, model_path: str = _MODEL_PATH) -> None:
    face_recognizer.write(model_path)


def load_model(model_path: str = _MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read(model_path)
    return face_recognizer


def draw_rect(test_img: np.ndarray, face: Rect, color: Tuple[int, int, int] = (255, 0, 0)) -> None:
    x, y, w, h = face
    cv2.rectangle(test_img, (x, y), (x + w, y + h), color, thickness=2)


def put_text(
    test_img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int] = (255, 0, 0),
) -> None:
    text_y = max(30, y - 10)
    cv2.putText(test_img, text, (x, text_y), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
