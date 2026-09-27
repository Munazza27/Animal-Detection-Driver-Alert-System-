# Animal Detection & Driver Alert System (ADAS)

Real-time Edge Vision, Risk-Aware Road Threat Assessment & Driver Safety Analytics Dashboard.

## Key Features

1. **Animal Detection Engine (YOLOv8)**:
   - Recognizes road animals (Deer, Cattle, Dog, Horse, Sheep, Cat, Tiger, Lion, Leopard, etc.).
   - Adaptive **MSRCR** and **CLAHE** low-light enhancement for dark night driving conditions.
   - Computes horizontal zone (`LEFT` / `CENTER` / `RIGHT`) and area fraction as a collision proximity proxy.

2. **Risk Assessment & Voice Alerts**:
   - Risk classification: `LOW` (Green), `MEDIUM` (Amber), `HIGH` (Red).
   - Priority handling for Center road corridor incursions.
   - Audible driver warnings and text-to-speech voice alerts.

3. **Streamlit Live Demo Interface**:
   - Video playback with real-time bounding box overlay color-coded by risk.
   - Dynamic **Driver Alert HUD**:
     - 🔴 **HIGH RISK ALERT**: Danger warnings when animals are in the path or close.
     - 🟡 **MEDIUM RISK CAUTION**: Caution warnings for roadside animals.
     - 🟢 **LOW RISK / CLEAR**: Road corridor safe.
   - 3-Zone Road Corridor Indicator (`LEFT` | `CENTER` | `RIGHT`).
   - Interactive playback controls: Play/Pause, Timeline Scrubber slider, Playback speed (0.5x - 2.0x), and Live YOLOv8 inference mode.
   - In-browser audible alert chimes.

4. **BI-Style Analytics Panel**:
   - **Detections per Session**: Aggregate metrics and interactive incident timelines across drive sessions.
   - **Risk-Level Distribution**: Donut charts, stacked species breakdown, and corridor zone hazard charts.
   - **Day vs. Night Comparison**: Side-by-side KPI cards, species detection comparison across daylight vs night, and risk severity shift analysis.
   - **Data Export**: Filter and export incident logs to CSV or JSON.

---

## Folder Structure

```
Animal-Detection-Driver-Alert-System-/
├── app.py                      # Streamlit Live Demo & BI Analytics Dashboard
├── main.py                     # Video detection batch processing script
├── risk_logic.py               # Risk analyzer & alert generator engine
├── requirements.txt            # System dependencies
├── src/
│   ├── config.py               # Thresholds, class lists, zone boundaries
│   ├── detector.py             # AnimalDetector (YOLOv8 wrapper)
│   ├── preprocess.py           # CLAHE & MSRCR low-light enhancement
│   ├── tracker.py              # Temporal smoothing for presence events
│   ├── utils.py                # Drawing, geometry, and risk evaluation helpers
│   └── analytics.py            # BI analytics engine and multi-session store
├── data/                       # Test and demo video footage (.mp4)
└── output/                     # Annotated videos, JSON logs, and session databases
```

---

## Setup & Running the Dashboard

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Streamlit App
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## CLI Usage

### Running Detection on Video via Terminal
```bash
python main.py --video data/cep.mp4 --output output/annotated_output.mp4 --json output/detections.json
```

### Running Risk Analysis & Alert Synthesis
```bash
python risk_logic.py
```