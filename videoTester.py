from __future__ import annotations

import argparse
import time
from typing import Dict

import cv2

import faceRecognition as fr

# Update this map to match the integer labels used in trainingImages/.
LABEL_NAMES: Dict[int, str] = {
    0: "Hanhan",
    1: "Tank",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time webcam facial recognition with LBPH.")
    parser.add_argument("--model", default="trainingData.yml", help="Path to the trained LBPH model.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--threshold", type=float, default=45.0, help="Accept a prediction only if confidence <= threshold.")
    parser.add_argument(
        "--detect-scale",
        type=float,
        default=0.75,
        help="Scale factor used for face detection only; smaller values reduce CPU usage.",
    )
    parser.add_argument("--frame-width", type=int, default=1000, help="Display width.")
    parser.add_argument("--frame-height", type=int, default=700, help="Display height.")
    parser.add_argument(
        "--log-predictions",
        action="store_true",
        help="Print label/confidence values for each detected face.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    face_recognizer = fr.load_model(args.model)
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            faces_detected, gray_img = fr.faceDetection(frame, detection_scale=args.detect_scale)

            for face in faces_detected:
                x, y, w, h = face
                roi_gray = fr.extract_face_roi(gray_img, face)
                label, confidence = face_recognizer.predict(roi_gray)

                if args.log_predictions:
                    print("confidence:", confidence, "label:", label)

                fr.draw_rect(frame, face)

                if confidence <= args.threshold:
                    predicted_name = LABEL_NAMES.get(label, f"ID-{label}")
                    fr.put_text(frame, predicted_name, x, y)
                else:
                    fr.put_text(frame, "Unknown", x, y)

            resized_img = fr.resize_for_display(frame, args.frame_width, args.frame_height)
            cv2.imshow("LBPH face recognition", resized_img)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
