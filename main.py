"""
main.py
-------
Entry point for Person 1s module. Run detection on a video file,
save an annotated output video, and export a per-frame JSON log.
"""

import argparse
import json
import os
import time

import cv2

from src.detector import AnimalDetector
from src.utils import draw_detections
from src.tracker import smooth_detections
from src.stream import open_source, is_live_source, parse_source


def run_on_video(video_path: str, output_video_path: str, output_json_path: str, show: bool = False):
    source = parse_source(video_path)
    live = is_live_source(source)

    if not live and not os.path.exists(source):
        raise FileNotFoundError(
            f"Video not found: {source}\n"
            f"Drop a test clip into data/raw_videos/ or run "
            f"tests/make_synthetic_video.py to generate a dummy test video."
        )

    detector = AnimalDetector()

    cap, live = open_source(source)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open source: {source}\n"
            f"If this is a phone stream, check the phone and laptop are on "
            f"the same Wi-Fi/hotspot and the URL matches what the camera "
            f"app shows."
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video_path, fourcc, fps if not live else 15, (frame_w, frame_h))

    all_frame_logs = []
    frame_index = 0
    start_time = time.time()

    if live:
        print(f"Live source connected: {source}")
        print("Press Ctrl+C to stop and save results.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                if live:
                    print("Stream ended or lost connection.")
                break

            detections = detector.detect(frame)

            annotated_frame = draw_detections(frame.copy(), detections)
            writer.write(annotated_frame)

            if live and show:
                try:
                    cv2.imshow("Animal Detection - press q to quit", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    pass

            timestamp = time.time() - start_time if live else frame_index / fps
            all_frame_logs.append({
                "frame_index": frame_index,
                "timestamp_sec": round(timestamp, 3),
                "detections": detections,
            })

            frame_index += 1
            if frame_index % 30 == 0:
                if live:
                    elapsed = time.time() - start_time
                    print(f"{frame_index} frames | {frame_index/elapsed:.1f} FPS live")
                else:
                    print(f"Processed {frame_index} frames...")
    except KeyboardInterrupt:
        print("\nStopped by user.")

    cap.release()
    writer.release()
    if live and show:
        cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    print(f"\nDone. Processed {frame_index} frames in {elapsed:.1f}s ({frame_index / elapsed:.1f} FPS average).")
    print(f"Annotated video saved to: {output_video_path}")

    with open(output_json_path, "w") as f:
        json.dump(all_frame_logs, f, indent=2)
    print(f"Per-frame detection log saved to: {output_json_path}")

    events = smooth_detections(all_frame_logs, fps=fps)
    events_path = output_json_path.replace(".json", "_events.json")
    with open(events_path, "w") as f:
        json.dump(events, f, indent=2)
    print(f"Smoothed presence events saved to: {events_path}")
    print(f"({len(events)} distinct animal presence event(s) detected)")

    return all_frame_logs, events


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run animal detection on a video file.")
    parser.add_argument("--video", type=str, default="data/raw_videos/synthetic_test.mp4", help="Path to input video file")
    parser.add_argument("--output", type=str, default="output/annotated_output.mp4", help="Path to save annotated output video")
    parser.add_argument("--json", type=str, default="output/detections.json", help="Path to save per-frame detection JSON log")
    parser.add_argument("--show", action="store_true", help="Pop up a window showing the live feed (needs a display).")
    args = parser.parse_args()

    run_on_video(args.video, args.output, args.json, show=args.show)
