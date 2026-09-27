"""
tracker.py
----------
Smooths raw, flickering per-frame detections (from detector.py) into
stable "presence events" -- e.g. "a Deer was present from 0.0s to 0.8s"
instead of a noisy log that flickers detected/not-detected every few
frames due to partial occlusion (tree trunks, motion blur, brief
confidence dips).

This does NOT do true multi-object tracking (it can't tell two deer
of the same species apart, or assign persistent IDs) -- it only
answers "is this class of animal continuously present, allowing for
short gaps?" which is exactly what's needed to avoid an alert system
flickering on/off every time a frame briefly misses a detection.

This is intentionally handed off as a ready-to-use building block for
Person 2 (Risk Logic) -- it turns the raw per-frame JSON into
something risk classification and alerts can consume directly,
without Person 2 needing to write occlusion-gap-handling logic
themselves.

Usage:
    from src.tracker import smooth_detections
    events = smooth_detections(all_frame_logs, fps=30, frames_to_clear=10)
"""

from src.config import FRAMES_TO_CLEAR


def smooth_detections(frame_logs: list, fps: float, frames_to_clear: int = FRAMES_TO_CLEAR) -> list:
    """
    Convert raw per-frame detection logs into smoothed presence events.

    Args:
        frame_logs: list of frame dicts, each like:
            {"frame_index": int, "timestamp_sec": float, "detections": [...]}
            (this is exactly what main.py writes to detections.json)
        fps: frames per second of the source video (used to convert
            frames_to_clear into a time gap for reference)
        frames_to_clear: how many consecutive frames a class can be
            absent before we consider the animal "gone" (bridges
            brief occlusion / confidence dips)

    Returns:
        List of presence event dicts:
        {
            "class_name": "Deer",
            "start_frame": 0, "end_frame": 24,
            "start_time": 0.0, "end_time": 0.8,
            "duration_sec": 0.8,
            "max_confidence": 0.57,
            "frames_detected": 9,      # how many frames actually had a detection
            "frames_total": 25,        # span length including bridged gaps
            "last_zone": "LEFT",       # zone at the most recent detected frame
            "max_area_fraction": 0.051 # closest approach (bigger = closer)
        }
    """
    # Track open events per class: class_name -> event-in-progress dict
    open_events = {}
    closed_events = []

    for frame in frame_logs:
        frame_index = frame["frame_index"]
        timestamp = frame["timestamp_sec"]
        seen_this_frame = set()

        for det in frame["detections"]:
            class_name = det["class_name"]
            seen_this_frame.add(class_name)

            if class_name not in open_events:
                # Start a new presence event
                open_events[class_name] = {
                    "class_name": class_name,
                    "start_frame": frame_index,
                    "end_frame": frame_index,
                    "start_time": timestamp,
                    "end_time": timestamp,
                    "max_confidence": det["confidence"],
                    "frames_detected": 1,
                    "last_seen_frame": frame_index,
                    "last_zone": det["zone"],
                    "max_area_fraction": det["area_fraction"],
                }
            else:
                ev = open_events[class_name]
                ev["end_frame"] = frame_index
                ev["end_time"] = timestamp
                ev["frames_detected"] += 1
                ev["last_seen_frame"] = frame_index
                ev["last_zone"] = det["zone"]
                if det["confidence"] > ev["max_confidence"]:
                    ev["max_confidence"] = det["confidence"]
                if det["area_fraction"] > ev["max_area_fraction"]:
                    ev["max_area_fraction"] = det["area_fraction"]

        # Close out any open events that have exceeded the gap tolerance
        for class_name in list(open_events.keys()):
            if class_name in seen_this_frame:
                continue
            ev = open_events[class_name]
            frames_since_seen = frame_index - ev["last_seen_frame"]
            if frames_since_seen > frames_to_clear:
                ev["frames_total"] = ev["end_frame"] - ev["start_frame"] + 1
                ev["duration_sec"] = round(ev["end_time"] - ev["start_time"], 3)
                closed_events.append(ev)
                del open_events[class_name]

    # Close any events still open at the end of the video
    for ev in open_events.values():
        ev["frames_total"] = ev["end_frame"] - ev["start_frame"] + 1
        ev["duration_sec"] = round(ev["end_time"] - ev["start_time"], 3)
        closed_events.append(ev)

    return closed_events


if __name__ == "__main__":
    # Quick manual test using a tiny synthetic example so this file
    # can be sanity-checked on its own without a real video.
    sample_logs = [
        {"frame_index": 0, "timestamp_sec": 0.0, "detections": [
            {"class_name": "Deer", "confidence": 0.57, "zone": "CENTER", "area_fraction": 0.027}]},
        {"frame_index": 1, "timestamp_sec": 0.033, "detections": [
            {"class_name": "Deer", "confidence": 0.55, "zone": "CENTER", "area_fraction": 0.027}]},
        {"frame_index": 2, "timestamp_sec": 0.067, "detections": []},
        {"frame_index": 3, "timestamp_sec": 0.1, "detections": []},
        {"frame_index": 4, "timestamp_sec": 0.133, "detections": [
            {"class_name": "Deer", "confidence": 0.51, "zone": "LEFT", "area_fraction": 0.03}]},
        {"frame_index": 5, "timestamp_sec": 0.167, "detections": []},
        {"frame_index": 6, "timestamp_sec": 0.2, "detections": []},
        {"frame_index": 7, "timestamp_sec": 0.233, "detections": []},
        {"frame_index": 8, "timestamp_sec": 0.267, "detections": []},
        {"frame_index": 9, "timestamp_sec": 0.3, "detections": []},
        {"frame_index": 10, "timestamp_sec": 0.333, "detections": []},
        {"frame_index": 11, "timestamp_sec": 0.367, "detections": []},
        {"frame_index": 12, "timestamp_sec": 0.4, "detections": []},
        {"frame_index": 13, "timestamp_sec": 0.433, "detections": []},
        {"frame_index": 14, "timestamp_sec": 0.467, "detections": []},  # gap > 10 frames -> new event after this
        {"frame_index": 15, "timestamp_sec": 0.5, "detections": [
            {"class_name": "Deer", "confidence": 0.48, "zone": "LEFT", "area_fraction": 0.05}]},
    ]

    events = smooth_detections(sample_logs, fps=30, frames_to_clear=10)
    import json
    print(json.dumps(events, indent=2))