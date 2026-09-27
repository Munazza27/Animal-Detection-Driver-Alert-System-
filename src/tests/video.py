"""
make_synthetic_video.py
------------------------
Generates a tiny synthetic test video so you can confirm the whole
pipeline (read video -> detect -> draw -> save -> JSON export) runs
end-to-end, BEFORE you plug in real dashcam/road footage.

Note: this synthetic video will NOT actually trigger animal
detections (it's just a moving rectangle, not a real cow/dog), so
don't worry if detections.json comes back empty when you run this.
Its only purpose is to prove the code runs without errors.
Swap in a real road video (with real animals in it) to see actual
detections.

Usage:
    python tests/make_synthetic_video.py
"""

import cv2
import numpy as np
import os

OUTPUT_PATH = "data/raw_videos/synthetic_test.mp4"
WIDTH, HEIGHT = 640, 480
FPS = 20
DURATION_SEC = 5


def make_video():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))

    total_frames = FPS * DURATION_SEC
    for i in range(total_frames):
        frame = np.full((HEIGHT, WIDTH, 3), (40, 90, 40), dtype=np.uint8)  # road-ish green/gray bg

        # simple moving gray rectangle to simulate "something" in frame
        x = int((i / total_frames) * (WIDTH - 100))
        cv2.rectangle(frame, (x, 250), (x + 100, 350), (120, 120, 120), -1)

        cv2.putText(frame, "SYNTHETIC TEST FRAME - NOT A REAL ANIMAL", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        writer.write(frame)

    writer.release()
    print(f"Synthetic test video created at: {OUTPUT_PATH}")
    print("Run: python main.py --video data/raw_videos/synthetic_test.mp4")


if __name__ == "__main__":
    make_video()