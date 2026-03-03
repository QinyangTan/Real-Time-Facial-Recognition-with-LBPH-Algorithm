from __future__ import annotations

import argparse
import os
from typing import Dict

import cv2

import faceRecognition as fr

# Update this map to match your folder labels in trainingImages/.
LABEL_NAMES: Dict[int, str] = {
    0: "Hanhan",
    1: "Tank",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LBPH model and test it on a single image.")
    parser.add_argument("--test-image", default="TestImages/Tank.jpg", help="Path to the test image.")
    parser.add_argument("--train-dir", default="trainingImages", help="Path to the labeled training image directory.")
    parser.add_argument("--model", default="trainingData.yml", help="Path to save/load the trained model.")
    parser.add_argument("--threshold", type=float, default=45.0, help="Accept a prediction only if confidence <= threshold.")
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Force retraining even if a saved model already exists.",
    )
    return parser.parse_args()


def load_or_train_model(train_dir: str, model_path: str, retrain: bool):
    if not retrain and os.path.exists(model_path):
        print(f"Loading existing model: {model_path}")
        return fr.load_model(model_path)

    print(f"Training model from: {train_dir}")
    faces, face_ids = fr.labels_for_training_data(train_dir)
    recognizer = fr.train_classifier(faces, face_ids)
    fr.save_model(recognizer, model_path)
    print(f"Saved trained model to: {model_path}")
    return recognizer


def main() -> None:
    args = parse_args()

    test_img = cv2.imread(args.test_image)
    if test_img is None:
        raise FileNotFoundError(f"Could not read test image: {args.test_image}")

    face_recognizer = load_or_train_model(args.train_dir, args.model, args.retrain)
    faces_detected, gray_img = fr.faceDetection(test_img)
    print("faces_detected:", faces_detected)

    for face in faces_detected:
        x, y, w, h = face
        roi_gray = fr.extract_face_roi(gray_img, face)
        label, confidence = face_recognizer.predict(roi_gray)

        print("confidence:", confidence)
        print("label:", label)

        fr.draw_rect(test_img, face)

        if confidence <= args.threshold:
            predicted_name = LABEL_NAMES.get(label, f"ID-{label}")
            fr.put_text(test_img, predicted_name, x, y)
        else:
            fr.put_text(test_img, "Unknown", x, y)

    resized_img = cv2.resize(test_img, (1000, 1000), interpolation=cv2.INTER_AREA)
    cv2.imshow("LBPH face recognition", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
