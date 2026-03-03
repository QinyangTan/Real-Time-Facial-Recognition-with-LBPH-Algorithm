from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resize a training image dataset into a parallel output folder.")
    parser.add_argument("--input-dir", default="trainingImages", help="Source image directory.")
    parser.add_argument("--output-dir", default="resizedTrainingImages", help="Destination directory.")
    parser.add_argument("--width", type=int, default=100, help="Target width.")
    parser.add_argument("--height", type=int, default=100, help="Target height.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    count = 0

    for path, _, filenames in os.walk(input_dir):
        filenames = sorted(filenames)
        current_path = Path(path)
        relative_path = current_path.relative_to(input_dir)
        target_folder = output_dir / relative_path
        target_folder.mkdir(parents=True, exist_ok=True)

        for filename in filenames:
            if filename.startswith("."):
                continue

            img_path = current_path / filename
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Skipping unreadable image: {img_path}")
                continue

            resized_image = cv2.resize(
                img,
                (args.width, args.height),
                interpolation=cv2.INTER_AREA,
            )

            output_path = target_folder / filename
            cv2.imwrite(str(output_path), resized_image)
            count += 1
            print(f"Saved: {output_path}")

    print(f"Done. Resized {count} image(s).")


if __name__ == "__main__":
    main()
