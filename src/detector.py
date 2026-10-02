"""
detector.py
-----------
Wraps YOLOv8 (via ultralytics) for animal detection.
This is Person 1's main deliverable: given a video frame, return
a clean list of detections that Person 2 (risk logic) can consume
directly, without needing to know anything about YOLO internals.

Output format per detection (a plain dict, easy to serialize to JSON):
{
    "class_name": "cow",
    "confidence": 0.87,
    "x1": 120.0, "y1": 340.0, "x2": 410.0, "y2": 610.0,
    "zone": "CENTER",          # added by add_position_info()
    "area_fraction": 0.081     # added by add_position_info()
}
"""

from ultralytics import YOLO #type:ignore
import numpy as np

from src.config import (
    ANIMAL_CLASSES,
    CONFIDENCE_THRESHOLD,
    CONFIDENCE_THRESHOLD_NIGHT,
    VERY_DARK_BRIGHTNESS,
    MODEL_WEIGHTS,
    LEFT_ZONE_END,
    RIGHT_ZONE_START,
)
from src.utils import get_horizontal_zone, box_area_fraction
from src.preprocess import preprocess_frame, is_very_dark


class AnimalDetector:
    def __init__(self, weights: str = MODEL_WEIGHTS, confidence: float = CONFIDENCE_THRESHOLD,
                 night_confidence: float = CONFIDENCE_THRESHOLD_NIGHT):
        """
        Loads the YOLOv8 model once. Reuse the same instance across
        all frames of a video -- don't recreate it per frame.
        """
        self.model = YOLO(weights)
        self.confidence = confidence
        self.night_confidence = night_confidence
        # Build a reverse lookup: COCO class name -> True if it's an animal we track
        self.tracked_classes = set(ANIMAL_CLASSES.values())

    def detect(self, frame: np.ndarray, apply_preprocessing: bool = True) -> list:
        """
        Run detection on a single frame.

        Args:
            frame: BGR image (numpy array) from OpenCV
            apply_preprocessing: whether to run low-light enhancement first

        Returns:
            List of detection dicts (see module docstring for format),
            filtered to only the animal classes we care about.
        """
        # Use a lower confidence bar on very dark frames -- at night we'd
        # rather surface a shaky "possible animal" alert than stay silent.
        # See config.py CONFIDENCE_THRESHOLD_NIGHT for the reasoning.
        active_threshold = self.night_confidence if is_very_dark(frame, VERY_DARK_BRIGHTNESS) else self.confidence

        input_frame = preprocess_frame(frame) if apply_preprocessing else frame

        # IMPORTANT: ultralytics' model() call has its own internal
        # default confidence filter (0.25) that runs BEFORE our code
        # sees any results. Since our night threshold can go as low as
        # 0.12, we must explicitly pass a low conf here, or the model
        # silently discards low-confidence night detections before we
        # ever get a chance to apply our own threshold logic below.
        results = self.model(input_frame, conf=0.05, verbose=False)[0]

        detections = []
        frame_h, frame_w = frame.shape[:2]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]
            confidence = float(box.conf[0])

            if class_name not in self.tracked_classes:
                continue
            if confidence < active_threshold:
                continue

            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

            detection = {
                "class_name": class_name,
                "confidence": confidence,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            }
            detection.update(self._add_position_info(detection, frame_w, frame_h))
            detections.append(detection)

        return detections

    @staticmethod
    def _add_position_info(detection: dict, frame_w: int, frame_h: int) -> dict:
        """Compute horizontal zone (LEFT/CENTER/RIGHT) and area fraction."""
        center_x = (detection["x1"] + detection["x2"]) / 2.0
        zone = get_horizontal_zone(center_x, frame_w, LEFT_ZONE_END, RIGHT_ZONE_START)
        area_fraction = box_area_fraction(
            detection["x1"], detection["y1"], detection["x2"], detection["y2"],
            frame_w, frame_h
        )
        return {"zone": zone, "area_fraction": round(area_fraction, 4)}