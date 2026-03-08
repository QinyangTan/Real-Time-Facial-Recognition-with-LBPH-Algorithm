from __future__ import annotations

import argparse
from pathlib import Path
from queue import Full, Queue
from threading import Thread
import time

import cv2

import faceRecognition as fr


class FrameWriter:
    def __init__(self, queue_size: int, verbose: bool) -> None:
        self._queue: Queue[object] = Queue(maxsize=max(1, queue_size))
        self._thread = Thread(target=self._run, daemon=True)
        self._verbose = verbose
        self.saved_count = 0
        self.failed_count = 0

    def start(self) -> None:
        self._thread.start()

    def submit(self, output_path: Path, frame) -> bool:
        try:
            self._queue.put_nowait((output_path, frame.copy()))
        except Full:
            return False
        return True

    def close(self) -> None:
        self._queue.put(None)
        self._thread.join()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return

                output_path, frame = item
                if cv2.imwrite(str(output_path), frame):
                    self.saved_count += 1
                    if self._verbose:
                        print(f"Saved: {output_path}")
                else:
                    self.failed_count += 1
                    print(f"Failed to save: {output_path}")
            finally:
                self._queue.task_done()


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
    parser.add_argument(
        "--writer-queue-size",
        type=int,
        default=8,
        help="Maximum number of pending frame writes before new save requests are dropped.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each successful save and dropped write request.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    frame_index = 0
    scheduled_count = 0
    queued_count = 0
    dropped_count = 0
    save_every = max(1, args.save_every)
    writer = FrameWriter(queue_size=args.writer_queue_size, verbose=args.verbose)
    writer.start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            if frame_index % save_every == 0:
                output_path = output_dir / f"frame{scheduled_count}.jpg"
                scheduled_count += 1
                if writer.submit(output_path, frame):
                    queued_count += 1
                else:
                    dropped_count += 1
                    if args.verbose:
                        print(f"Dropped save request (writer queue full): {output_path}")

            frame_index += 1

            resized_img = fr.resize_for_display(frame, args.preview_width, args.preview_height)
            cv2.imshow("Frame capture", resized_img)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        writer.close()
        cv2.destroyAllWindows()
        print(
            "Capture ended. "
            f"Queued {queued_count} frame(s), "
            f"saved {writer.saved_count} frame(s), "
            f"dropped {dropped_count} frame(s), "
            f"failed {writer.failed_count} frame(s)."
        )


if __name__ == "__main__":
    main()
