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


def evaluate_detection_risk(det: dict, duration_sec: float = 0.0) -> tuple:
    """
    Evaluate the risk level and driver alert for a single detection dictionary.
    Returns: (risk_level: str, alert_message: str)
    """
    area_fraction = det.get("area_fraction", 0.0)
    zone = det.get("zone", "CENTER")
    animal = det.get("class_name", "Animal")

    # Risk based on animal size proxy in frame
    if area_fraction >= 0.15:
        risk = "High"
    elif area_fraction >= 0.08:
        risk = "Medium"
    else:
        risk = "Low"

    # Center zone is direct collision corridor
    if zone == "CENTER":
        if risk == "Low":
            risk = "Medium"
        elif risk == "Medium":
            risk = "High"

    # Persistent duration increases risk
    if duration_sec >= 2.0 and risk == "Low":
        risk = "Medium"

    # Formulate alert text
    if risk == "High":
        if zone == "CENTER":
            alert = f"Warning, {animal} ahead. Slow down."
        else:
            alert = f"Warning, {animal} on the {zone.lower()}. Slow down."
    elif risk == "Medium":
        if zone == "CENTER":
            alert = f"Caution, {animal} ahead."
        else:
            alert = f"Caution, {animal} on the {zone.lower()}."
    else:
        alert = f"{animal} detected on {zone.lower()} (Low Risk)."

    return risk, alert


def draw_detections(frame: np.ndarray, detections: list, draw_lanes: bool = False,
                    left_zone_end: float = 0.35, right_zone_start: float = 0.65) -> np.ndarray:
    """
    Draw bounding boxes + labels for a list of detection dicts on the frame.
    Each detection dict is expected to have:
        x1, y1, x2, y2, class_name, confidence, zone (optional), risk (optional)

    Box color changes with risk level for quick visual read:
        LOW -> Green (0, 200, 0)
        MEDIUM -> Amber/Yellow (0, 200, 255)
        HIGH -> Bright Red (0, 0, 255)
    """
    color_map = {
        "LOW": (0, 200, 0),        # Green (BGR)
        "MEDIUM": (0, 200, 255),    # Amber/Yellow (BGR)
        "HIGH": (0, 0, 255),        # Red (BGR)
    }

    h, w = frame.shape[:2]

    # Optional HUD zone divider lines
    if draw_lanes:
        x_left = int(w * left_zone_end)
        x_right = int(w * right_zone_start)
        cv2.line(frame, (x_left, 0), (x_left, h), (70, 70, 70), 1, cv2.LINE_AA)
        cv2.line(frame, (x_right, 0), (x_right, h), (70, 70, 70), 1, cv2.LINE_AA)

    for det in detections:
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])

        # Determine risk if not pre-assigned
        if "risk" in det and det["risk"]:
            risk_raw = det["risk"]
        else:
            risk_raw, _ = evaluate_detection_risk(det)
            det["risk"] = risk_raw

        risk_upper = str(risk_raw).upper()
        color = color_map.get(risk_upper, (0, 200, 0))

        # Main bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Corner accents for sleek modern HUD look
        corner_len = min(15, max(6, int((x2 - x1) * 0.15)))
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, 4)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, 4)
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, 4)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, 4)
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, 4)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, 4)
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, 4)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, 4)

        label = f'{det["class_name"]} {det["confidence"]:.2f}'
        if "zone" in det:
            label += f' | {det["zone"]}'
        label += f' | {risk_upper}'

        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - text_h - 10)), (x1 + text_w + 8, y1), color, -1)
        cv2.putText(frame, label, (x1 + 4, max(text_h + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame