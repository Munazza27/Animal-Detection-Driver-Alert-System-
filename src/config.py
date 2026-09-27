"""
config.py
---------
Central place for all tunable settings: which classes count as
"animals", confidence thresholds, and risk-zone boundaries.

Person 2 (Risk Logic) will read RISK_ZONES and CLASS thresholds
from here too, so keep this file as the single source of truth.
"""

# COCO class names that YOLOv8 (pretrained) already recognizes.
# We only care about a subset that maps to real Indian road animals.
# COCO id -> name (for reference, ultralytics gives us names directly)
ANIMAL_CLASSES = {
    "Deer": "Deer",
    "Antelope": "Antelope",
    "Tiger": "Tiger",
    "Leopard": "Leopard",
    "Lion": "Lion",
    "Lynx": "Lynx",
    "Cheetah": "Cheetah",
    "Fox": "Fox",
    "Camel": "Camel",
    "Cattle": "Cattle",
    "Dog": "Dog",
    "Horse": "Horse",
    "Sheep": "Sheep",
    "Cat": "Cat",          # feline family (tiger/leopard) often misread as this in low light
    "Animal": "Animal",   # generic fallback for anything else
}

# Minimum confidence to accept a detection at all (normal/daylight frames)
CONFIDENCE_THRESHOLD = 0.35

# Lower threshold used specifically on very dark frames -- see
# src/preprocess.py's is_very_dark(). At night we'd rather surface a
# shaky "possible animal" alert than stay silent; false positives are
# a minor annoyance, false negatives are a safety risk.
# NOTE: tested against real near-black footage -- genuine animal signals
# in this condition land around 0.10-0.17 confidence, so 0.35 (the
# daylight threshold) would miss them entirely. 0.12 is a deliberately
# permissive floor for this reason; expect more false positives at
# night as the direct trade-off.
CONFIDENCE_THRESHOLD_NIGHT = 0.12

# Brightness (0-255 mean pixel value) below which a frame is treated
# as "very dark" and gets the lower night threshold + MSRCR enhancement
VERY_DARK_BRIGHTNESS = 55.0

# Model weights to use. Open Images V7 (601 classes) instead of COCO (80)
# -- gives us real Deer/Tiger/Leopard classes instead of nearest-COCO-match guesses.
# Using the "small" variant (not nano) -- confirmed via testing that it
# catches weak night-time signals nano misses entirely.
MODEL_WEIGHTS = "yolov8s-oiv7.pt"

# Frame is divided into 3 horizontal zones for left/center/right positioning.
# Values are fractions of frame width (0.0 - 1.0)
LEFT_ZONE_END = 0.35
RIGHT_ZONE_START = 0.65

# Bounding box area (as a fraction of frame area) used as a rough
# "distance proxy" -- bigger box fraction = animal is closer.
# These thresholds are a starting point; Person 2 should tune them
# against real test footage and document the reasoning.
RISK_AREA_THRESHOLDS = {
    "LOW": 0.02,      # box occupies < 2% of frame -> far away
    "MEDIUM": 0.06,   # 2% - 6% of frame -> approaching
    "HIGH": 0.06,     # > 6% of frame -> close / in path
}

# How many consecutive frames an animal must be absent before we
# consider it "gone" (avoids flicker causing repeated alerts)
FRAMES_TO_CLEAR = 10