# Animal Detection Model

Detection engine for the AI-Based Animal Detection & Driver Alert System.
This module takes video input and outputs, per frame, the list of animals
detected (class, confidence, bounding box, screen position, and a distance
proxy)  ready for **Person 2 (Risk Logic)** to consume directly.

## What this module does
- Loads a pretrained **YOLOv8** model (trained on COCO no training required)
- Filters detections down to real-world road animal classes: `cow`, `horse`,
  `sheep`, `dog`, `cat`, `bird`, `elephant`, `bear`, `zebra`, `giraffe`
- Applies **CLAHE low-light enhancement** automatically on dark frames
  (night driving conditions)
- Computes, per detection:
  - horizontal **zone**: `LEFT` / `CENTER` / `RIGHT`
  - **area_fraction**: how much of the frame the bounding box occupies
    (used as a rough distance proxy — bigger = closer)
- Outputs:
  - an annotated video with bounding boxes drawn
  - a `detections.json` file with a full per-frame log

## Folder structure
```
animal-detection-model/
├── main.py                     # entry point — run detection on a video
├── requirements.txt
├── src/
│   ├── config.py                # thresholds, class list, zone boundaries
│   ├── detector.py               # AnimalDetector class (YOLOv8 wrapper)
│   ├── preprocess.py             # CLAHE low-light enhancement
│   └── utils.py                  # drawing + geometry helper functions
├── tests/
│   └── make_synthetic_video.py   # generates a dummy video to sanity-check the pipeline
├── data/
│   └── raw_videos/               # put your test/demo footage here (not tracked by git)
└── output/                       # annotated video + detections.json land here
```

## Setup
```bash
pip install -r requirements.txt
```
(First run will auto-download `yolov8n.pt`, ~6MB — this is expected.)

## Quick pipeline check (no real footage needed)
Before you have real road footage, confirm the whole pipeline runs without errors:
```bash
python tests/make_synthetic_video.py
python main.py --video data/raw_videos/synthetic_test.mp4
```
This won't produce real detections (it's just a moving rectangle, not an
actual animal) — it only proves the code path works end to end.

## Running on real footage
Drop your video(s) into `data/raw_videos/`, then:
```bash
python main.py --video data/raw_videos/your_clip.mp4 --output output/annotated.mp4 --json output/detections.json
```

## Output format (`detections.json`)
```json
[
  {
    "frame_index": 42,
    "timestamp_sec": 1.4,
    "detections": [
      {
        "class_name": "cow",
        "confidence": 0.87,
        "x1": 120.0, "y1": 340.0, "x2": 410.0, "y2": 610.0,
        "zone": "CENTER",
        "area_fraction": 0.081
      }
    ]
  }
]
```
Person 2 reads this file (or calls `AnimalDetector.detect()` directly frame-by-frame
for a live pipeline) and turns `zone` + `area_fraction` into risk levels
(Low/Medium/High) and voice alerts.

## Notes on accuracy / next steps (for documentation)
- Current model is **pretrained only** (COCO) — no fine-tuning done.
- For production-grade accuracy on Indian-specific cases (e.g. stray cattle
  at unusual angles, breeds not well represented in COCO), fine-tuning on
  Indian road datasets (DATS_2022, IDD, Roboflow Indian Roads Detection) is
  the proposed next step — see project documentation for details.
- Confidence threshold and risk-area thresholds are defined in `src/config.py`
  and should be tuned against real test footage; log your reasoning for the
  documentation/results section.