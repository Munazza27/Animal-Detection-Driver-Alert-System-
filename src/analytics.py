"""
analytics.py
------------
Data storage and aggregation engine for the BI Analytics Panel.
Aggregates detection sessions, risk metrics, and Day vs Night comparisons.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional

SESSIONS_DB_PATH = "output/analytics_sessions.json"


def compute_session_metrics(session_id: str,
                            session_name: str,
                            video_name: str,
                            lighting: str,
                            mean_brightness: float,
                            fps: float,
                            frame_logs: List[Dict[str, Any]],
                            events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes summary metrics for a given driving session.
    """
    total_frames = len(frame_logs)
    duration_sec = round(total_frames / fps, 2) if fps > 0 else 0.0

    # Aggregate frame-level detections
    total_detections = 0
    detections_by_class = {}
    detections_by_zone = {"LEFT": 0, "CENTER": 0, "RIGHT": 0}
    confidences = []
    
    for frame in frame_logs:
        dets = frame.get("detections", [])
        total_detections += len(dets)
        for d in dets:
            c_name = d.get("class_name", "Animal")
            detections_by_class[c_name] = detections_by_class.get(c_name, 0) + 1
            z = d.get("zone", "CENTER")
            detections_by_zone[z] = detections_by_zone.get(z, 0) + 1
            confidences.append(d.get("confidence", 0.5))

    # Aggregate event-level risk metrics
    risk_counts = {"Low": 0, "Medium": 0, "High": 0}
    events_list = []
    for ev in events:
        risk = ev.get("risk")
        if not risk:
            # compute risk if not already present
            area = ev.get("max_area_fraction", 0.04)
            zone = ev.get("last_zone", "CENTER")
            duration = ev.get("duration_sec", 0.5)
            if area >= 0.15:
                risk = "High"
            elif area >= 0.08:
                risk = "Medium"
            else:
                risk = "Low"
            if zone == "CENTER":
                if risk == "Low":
                    risk = "Medium"
                elif risk == "Medium":
                    risk = "High"
            if duration >= 2.0 and risk == "Low":
                risk = "Medium"
        else:
            risk = risk.capitalize()

        risk_counts[risk] = risk_counts.get(risk, 0) + 1
        
        # generate alert text if missing
        animal = ev.get("class_name", ev.get("animal", "Animal"))
        zone = ev.get("last_zone", ev.get("position", "CENTER"))
        alert = ev.get("alert")
        if not alert:
            if risk == "High":
                alert = f"Warning, {animal} ahead. Slow down." if zone == "CENTER" else f"Warning, {animal} on the {zone.lower()}. Slow down."
            elif risk == "Medium":
                alert = f"Caution, {animal} ahead." if zone == "CENTER" else f"Caution, {animal} on the {zone.lower()}."
            else:
                alert = None

        events_list.append({
            "animal": animal,
            "start_time": ev.get("start_time", 0.0),
            "end_time": ev.get("end_time", 0.0),
            "duration_sec": ev.get("duration_sec", round(ev.get("end_time", 0) - ev.get("start_time", 0), 2)),
            "confidence": round(ev.get("max_confidence", ev.get("confidence", 0.7)), 3),
            "position": zone,
            "area_fraction": round(ev.get("max_area_fraction", ev.get("area_fraction", 0.05)), 4),
            "risk": risk,
            "alert": alert,
            "lighting": lighting,
        })

    avg_conf = round(float(np.mean(confidences)), 3) if confidences else 0.0

    return {
        "session_id": session_id,
        "session_name": session_name,
        "video_name": video_name,
        "lighting": lighting,
        "mean_brightness": round(mean_brightness, 1),
        "duration_sec": duration_sec,
        "total_frames": total_frames,
        "total_detections": total_detections,
        "total_events": len(events_list),
        "avg_confidence": avg_conf,
        "risk_counts": risk_counts,
        "detections_by_class": detections_by_class,
        "detections_by_zone": detections_by_zone,
        "events": events_list
    }


def get_default_sessions() -> List[Dict[str, Any]]:
    """
    Constructs multi-session driving data including existing output from cep.mp4
    plus representative Day, Night, and Twilight sessions for comprehensive BI analytics.
    """
    sessions = []

    # 1. Existing Session from output/detections.json & output/risk_events.json (cep.mp4)
    cep_events = []
    if os.path.exists("output/risk_events.json"):
        with open("output/risk_events.json", "r") as f:
            cep_events = json.load(f)
    elif os.path.exists("output/detections_events.json"):
        with open("output/detections_events.json", "r") as f:
            cep_events = json.load(f)

    cep_frames = []
    if os.path.exists("output/detections.json"):
        with open("output/detections.json", "r") as f:
            cep_frames = json.load(f)

    session_1 = compute_session_metrics(
        session_id="session_01",
        session_name="Session 1: Forest Corridor (Night)",
        video_name="cep.mp4",
        lighting="Night",
        mean_brightness=41.2,
        fps=30.0,
        frame_logs=cep_frames,
        events=cep_events
    )
    sessions.append(session_1)

    # 2. Session 2: Rural Highway (Daylight) - Cattle Crossing
    session_2_events = [
        {"class_name": "Cattle", "start_time": 2.1, "end_time": 6.8, "duration_sec": 4.7, "max_confidence": 0.89, "last_zone": "CENTER", "max_area_fraction": 0.18, "risk": "High", "alert": "Warning, Cattle ahead. Slow down."},
        {"class_name": "Cattle", "start_time": 3.0, "end_time": 7.2, "duration_sec": 4.2, "max_confidence": 0.84, "last_zone": "LEFT", "max_area_fraction": 0.12, "risk": "Medium", "alert": "Caution, Cattle on the left."},
        {"class_name": "Dog", "start_time": 9.4, "end_time": 11.0, "duration_sec": 1.6, "max_confidence": 0.78, "last_zone": "RIGHT", "max_area_fraction": 0.04, "risk": "Low", "alert": None},
        {"class_name": "Horse", "start_time": 14.5, "end_time": 18.2, "duration_sec": 3.7, "max_confidence": 0.92, "last_zone": "CENTER", "max_area_fraction": 0.16, "risk": "High", "alert": "Warning, Horse ahead. Slow down."},
        {"class_name": "Sheep", "start_time": 22.0, "end_time": 25.5, "duration_sec": 3.5, "max_confidence": 0.81, "last_zone": "LEFT", "max_area_fraction": 0.07, "risk": "Medium", "alert": "Caution, Sheep on the left."},
        {"class_name": "Dog", "start_time": 28.1, "end_time": 29.8, "duration_sec": 1.7, "max_confidence": 0.75, "last_zone": "RIGHT", "max_area_fraction": 0.03, "risk": "Low", "alert": None}
    ]
    # Synthetic frame representation for session 2
    s2_frames = []
    for f_idx in range(900): # 30 sec @ 30fps
        sec = f_idx / 30.0
        dets = []
        if 2.1 <= sec <= 6.8:
            dets.append({"class_name": "Cattle", "confidence": 0.89, "zone": "CENTER", "area_fraction": 0.18})
        if 3.0 <= sec <= 7.2:
            dets.append({"class_name": "Cattle", "confidence": 0.84, "zone": "LEFT", "area_fraction": 0.12})
        if 9.4 <= sec <= 11.0:
            dets.append({"class_name": "Dog", "confidence": 0.78, "zone": "RIGHT", "area_fraction": 0.04})
        if 14.5 <= sec <= 18.2:
            dets.append({"class_name": "Horse", "confidence": 0.92, "zone": "CENTER", "area_fraction": 0.16})
        if 22.0 <= sec <= 25.5:
            dets.append({"class_name": "Sheep", "confidence": 0.81, "zone": "LEFT", "area_fraction": 0.07})
        s2_frames.append({"frame_index": f_idx, "timestamp_sec": sec, "detections": dets})

    session_2 = compute_session_metrics(
        session_id="session_02",
        session_name="Session 2: Rural Highway (Day)",
        video_name="highway_day_drive.mp4",
        lighting="Day",
        mean_brightness=162.5,
        fps=30.0,
        frame_logs=s2_frames,
        events=session_2_events
    )
    sessions.append(session_2)

    # 3. Session 3: Suburban Twilight Drive - Stray Animals
    session_3_events = [
        {"class_name": "Dog", "start_time": 1.5, "end_time": 4.2, "duration_sec": 2.7, "max_confidence": 0.72, "last_zone": "CENTER", "max_area_fraction": 0.09, "risk": "Medium", "alert": "Caution, Dog ahead."},
        {"class_name": "Cat", "start_time": 6.0, "end_time": 7.4, "duration_sec": 1.4, "max_confidence": 0.65, "last_zone": "LEFT", "max_area_fraction": 0.04, "risk": "Low", "alert": None},
        {"class_name": "Dog", "start_time": 10.2, "end_time": 14.5, "duration_sec": 4.3, "max_confidence": 0.83, "last_zone": "CENTER", "max_area_fraction": 0.15, "risk": "High", "alert": "Warning, Dog ahead. Slow down."},
        {"class_name": "Cattle", "start_time": 16.0, "end_time": 19.8, "duration_sec": 3.8, "max_confidence": 0.77, "last_zone": "RIGHT", "max_area_fraction": 0.11, "risk": "Medium", "alert": "Caution, Cattle on the right."}
    ]
    s3_frames = []
    for f_idx in range(600): # 20 sec @ 30fps
        sec = f_idx / 30.0
        dets = []
        if 1.5 <= sec <= 4.2:
            dets.append({"class_name": "Dog", "confidence": 0.72, "zone": "CENTER", "area_fraction": 0.09})
        if 6.0 <= sec <= 7.4:
            dets.append({"class_name": "Cat", "confidence": 0.65, "zone": "LEFT", "area_fraction": 0.04})
        if 10.2 <= sec <= 14.5:
            dets.append({"class_name": "Dog", "confidence": 0.83, "zone": "CENTER", "area_fraction": 0.15})
        if 16.0 <= sec <= 19.8:
            dets.append({"class_name": "Cattle", "confidence": 0.77, "zone": "RIGHT", "area_fraction": 0.11})
        s3_frames.append({"frame_index": f_idx, "timestamp_sec": sec, "detections": dets})

    session_3 = compute_session_metrics(
        session_id="session_03",
        session_name="Session 3: Suburban Bypass (Twilight)",
        video_name="suburban_twilight.mp4",
        lighting="Night", # twilight/dim treated as low-light
        mean_brightness=52.8,
        fps=30.0,
        frame_logs=s3_frames,
        events=session_3_events
    )
    sessions.append(session_3)

    # 4. Session 4: Forest Wildlife Sanctuary (Day) - Deer & Wild Animals
    session_4_events = [
        {"class_name": "Deer", "start_time": 3.2, "end_time": 7.5, "duration_sec": 4.3, "max_confidence": 0.88, "last_zone": "CENTER", "max_area_fraction": 0.17, "risk": "High", "alert": "Warning, Deer ahead. Slow down."},
        {"class_name": "Deer", "start_time": 8.0, "end_time": 10.5, "duration_sec": 2.5, "max_confidence": 0.79, "last_zone": "RIGHT", "max_area_fraction": 0.08, "risk": "Medium", "alert": "Caution, Deer on the right."},
        {"class_name": "Fox", "start_time": 13.1, "end_time": 14.8, "duration_sec": 1.7, "max_confidence": 0.68, "last_zone": "LEFT", "max_area_fraction": 0.03, "risk": "Low", "alert": None},
        {"class_name": "Antelope", "start_time": 18.0, "end_time": 21.2, "duration_sec": 3.2, "max_confidence": 0.85, "last_zone": "CENTER", "max_area_fraction": 0.14, "risk": "High", "alert": "Warning, Antelope ahead. Slow down."}
    ]
    s4_frames = []
    for f_idx in range(750): # 25 sec @ 30fps
        sec = f_idx / 30.0
        dets = []
        if 3.2 <= sec <= 7.5:
            dets.append({"class_name": "Deer", "confidence": 0.88, "zone": "CENTER", "area_fraction": 0.17})
        if 8.0 <= sec <= 10.5:
            dets.append({"class_name": "Deer", "confidence": 0.79, "zone": "RIGHT", "area_fraction": 0.08})
        if 13.1 <= sec <= 14.8:
            dets.append({"class_name": "Fox", "confidence": 0.68, "zone": "LEFT", "area_fraction": 0.03})
        if 18.0 <= sec <= 21.2:
            dets.append({"class_name": "Antelope", "confidence": 0.85, "zone": "CENTER", "area_fraction": 0.14})
        s4_frames.append({"frame_index": f_idx, "timestamp_sec": sec, "detections": dets})

    session_4 = compute_session_metrics(
        session_id="session_04",
        session_name="Session 4: Wildlife Sanctuary (Day)",
        video_name="sanctuary_day.mp4",
        lighting="Day",
        mean_brightness=148.0,
        fps=30.0,
        frame_logs=s4_frames,
        events=session_4_events
    )
    sessions.append(session_4)

    # 5. Session 5: Expressway Night Hazard (Night) - Dark Corridor
    session_5_events = [
        {"class_name": "Cattle", "start_time": 4.5, "end_time": 8.9, "duration_sec": 4.4, "max_confidence": 0.74, "last_zone": "CENTER", "max_area_fraction": 0.19, "risk": "High", "alert": "Warning, Cattle ahead. Slow down."},
        {"class_name": "Dog", "start_time": 12.0, "end_time": 14.2, "duration_sec": 2.2, "max_confidence": 0.61, "last_zone": "LEFT", "max_area_fraction": 0.06, "risk": "Medium", "alert": "Caution, Dog on the left."},
        {"class_name": "Animal", "start_time": 16.5, "end_time": 18.0, "duration_sec": 1.5, "max_confidence": 0.45, "last_zone": "RIGHT", "max_area_fraction": 0.03, "risk": "Low", "alert": None}
    ]
    s5_frames = []
    for f_idx in range(600):
        sec = f_idx / 30.0
        dets = []
        if 4.5 <= sec <= 8.9:
            dets.append({"class_name": "Cattle", "confidence": 0.74, "zone": "CENTER", "area_fraction": 0.19})
        if 12.0 <= sec <= 14.2:
            dets.append({"class_name": "Dog", "confidence": 0.61, "zone": "LEFT", "area_fraction": 0.06})
        if 16.5 <= sec <= 18.0:
            dets.append({"class_name": "Animal", "confidence": 0.45, "zone": "RIGHT", "area_fraction": 0.03})
        s5_frames.append({"frame_index": f_idx, "timestamp_sec": sec, "detections": dets})

    session_5 = compute_session_metrics(
        session_id="session_05",
        session_name="Session 5: Expressway Bypass (Night)",
        video_name="expressway_night.mp4",
        lighting="Night",
        mean_brightness=28.4,
        fps=30.0,
        frame_logs=s5_frames,
        events=session_5_events
    )
    sessions.append(session_5)

    return sessions


def load_all_sessions() -> List[Dict[str, Any]]:
    """
    Loads sessions from JSON file. If file does not exist, initializes with defaults.
    """
    os.makedirs(os.path.dirname(SESSIONS_DB_PATH), exist_ok=True)
    if os.path.exists(SESSIONS_DB_PATH):
        try:
            with open(SESSIONS_DB_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass

    sessions = get_default_sessions()
    save_sessions(sessions)
    return sessions


def save_sessions(sessions: List[Dict[str, Any]]) -> None:
    """Save sessions back to JSON database."""
    os.makedirs(os.path.dirname(SESSIONS_DB_PATH), exist_ok=True)
    with open(SESSIONS_DB_PATH, "w") as f:
        json.dump(sessions, f, indent=2)


def add_or_update_session(new_session: Dict[str, Any]) -> None:
    """Add a new session or update an existing one."""
    sessions = load_all_sessions()
    idx = next((i for i, s in enumerate(sessions) if s["session_id"] == new_session["session_id"]), None)
    if idx is not None:
        sessions[idx] = new_session
    else:
        sessions.append(new_session)
    save_sessions(sessions)


def get_all_events_df(sessions: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
    """
    Extracts all events from all sessions into a flattened pandas DataFrame for BI plotting.
    """
    if sessions is None:
        sessions = load_all_sessions()

    rows = []
    for s in sessions:
        s_id = s.get("session_id")
        s_name = s.get("session_name")
        s_light = s.get("lighting")
        s_brightness = s.get("mean_brightness", 50.0)

        for ev in s.get("events", []):
            rows.append({
                "session_id": s_id,
                "session_name": s_name,
                "lighting": s_light,
                "brightness": s_brightness,
                "animal": ev.get("animal", "Animal"),
                "risk": ev.get("risk", "Low"),
                "position": ev.get("position", "CENTER"),
                "duration_sec": ev.get("duration_sec", 1.0),
                "confidence": ev.get("confidence", 0.5),
                "area_fraction": ev.get("area_fraction", 0.05),
                "alert": ev.get("alert", "None"),
                "start_time": ev.get("start_time", 0.0),
                "end_time": ev.get("end_time", 0.0),
            })

    return pd.DataFrame(rows)


def get_sessions_summary_df(sessions: Optional[List[Dict[str, Any]]] = None) -> pd.DataFrame:
    """
    Generates high-level summary DataFrame across all sessions.
    """
    if sessions is None:
        sessions = load_all_sessions()

    rows = []
    for s in sessions:
        rc = s.get("risk_counts", {})
        rows.append({
            "session_id": s["session_id"],
            "session_name": s["session_name"],
            "lighting": s["lighting"],
            "mean_brightness": s.get("mean_brightness", 50.0),
            "duration_sec": s.get("duration_sec", 0.0),
            "total_detections": s.get("total_detections", 0),
            "total_events": s.get("total_events", 0),
            "high_risk": rc.get("High", 0),
            "medium_risk": rc.get("Medium", 0),
            "low_risk": rc.get("Low", 0),
            "avg_confidence": s.get("avg_confidence", 0.0)
        })

    return pd.DataFrame(rows)
