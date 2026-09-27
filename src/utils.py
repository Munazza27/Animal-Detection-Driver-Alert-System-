"""
utils.py
--------
Small helper functions used across the detection pipeline:
drawing boxes on frames, computing box position/area, etc.
 
Person 3 (Web Interface) can reuse draw_detections() directly
if they want the same visual style in the dashboard.
"""
 
import cv2
import numpy as np
 
 
def box_center_x(x1: float, x2: float) -> float:
    """Return the horizontal center of a bounding box."""
    return (x1 + x2) / 2.0
 
 
def box_area_fraction(x1, y1, x2, y2, frame_width, frame_height) -> float:
    """
    Return what fraction of the total frame area this bounding
    box occupies. Used as a rough "how close is it" proxy.
    """
    box_area = max(0, (x2 - x1)) * max(0, (y2 - y1))
    frame_area = frame_width * frame_height
    if frame_area == 0:
        return 0.0
    return box_area / frame_area
 
 
def get_horizontal_zone(center_x: float, frame_width: int, left_end: float, right_start: float) -> str:
    """
    Classify a point's horizontal position into left/center/right
    based on fractional zone boundaries (0.0 - 1.0 of frame width).
    """
    fraction = center_x / frame_width if frame_width else 0.5
    if fraction < left_end:
        return "LEFT"
    elif fraction > right_start:
        return "RIGHT"
    else:
        return "CENTER"
 
 
def draw_detections(frame: np.ndarray, detections: list) -> np.ndarray:
    """
    Draw bounding boxes + labels for a list of detection dicts on the frame.
    Each detection dict is expected to have:
        x1, y1, x2, y2, class_name, confidence, zone (optional), risk (optional)
 
    Box color changes with risk level for quick visual read:
        LOW -> green, MEDIUM -> yellow, HIGH -> red, unknown -> blue
    """
    color_map = {
        "LOW": (0, 200, 0),
        "MEDIUM": (0, 200, 255),
        "HIGH": (0, 0, 255),
    }
 
    for det in detections:
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        risk = det.get("risk", "LOW")
        color = color_map.get(risk, (255, 128, 0))
 
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
 
        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        if "zone" in det:
            label += f' | {det["zone"]}'
        if "risk" in det:
            label += f' | {risk}'
 
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
 
    return frame
 