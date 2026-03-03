from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture webcam frames and save them as JPG images.")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index.")
    parser.add_argument("--output-dir", default="capturedFrames", help="Folder for saved frames.")
    parser.add_argument(
        "--save-every",
        type=int,
        default=8,
        help="Save one frame every N frames to reduce duplicates.",
    )
    parser.add_argument("--preview-width", type=int, default=1000, help="Preview width.")
    parser.add_argument("--preview-height", type=int, default=700, help="Preview height.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    frame_index = 0
    saved_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            if frame_index % max(1, args.save_every) == 0:
                output_path = output_dir / f"frame{saved_count}.jpg"
                cv2.imwrite(str(output_path), frame)
                print(f"Saved: {output_path}")
                saved_count += 1

            frame_index += 1

            resized_img = cv2.resize(
                frame,
                (args.preview_width, args.preview_height),
                interpolation=cv2.INTER_AREA,
            )
            cv2.imshow("Frame capture", resized_img)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
