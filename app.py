"""
app.py
------
Animal Detection & Driver Alert System (ADAS)
- Live Driver HUD: Video stream & live camera with clean risk overlay & corridor tracking
- BI Analytics Panel: Sessions, risk distribution, and Day vs. Night comparisons
- Session Manager: Batch processing and analytics synchronization
"""

import os
import time
import json
import glob
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Local imports
from src.config import (
    ANIMAL_CLASSES,
    CONFIDENCE_THRESHOLD,
    CONFIDENCE_THRESHOLD_NIGHT,
    VERY_DARK_BRIGHTNESS,
    LEFT_ZONE_END,
    RIGHT_ZONE_START,
)
from src.utils import draw_detections, evaluate_detection_risk
from src.preprocess import is_very_dark, is_low_light
from src.analytics import (
    load_all_sessions,
    get_all_events_df,
    get_sessions_summary_df,
    compute_session_metrics,
    add_or_update_session,
)
from src.sound_alert import play_sound_alert

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SafeDrive ADAS | Animal Detection & Driver Alert",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# MINIMALIST & CLEAN CSS DESIGN SYSTEM
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Clean layout & typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 96%;
    }
    
    /* Clean Header Banner */
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 20px;
        background: #0f172a;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        margin-bottom: 20px;
    }
    .header-title-box h1 {
        font-size: 1.45rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .header-subtitle {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 3px;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.78rem;
        font-weight: 600;
        padding: 5px 12px;
        border-radius: 20px;
        background: rgba(16, 185, 129, 0.12);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #10b981;
    }

    /* Universal Clean Cards */
    .clean-card {
        background: #111827;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 14px;
    }
    
    /* Alert Status Cards */
    .alert-card {
        border-radius: 12px;
        padding: 16px 18px;
        margin-bottom: 14px;
        transition: all 0.2s ease;
    }
    .alert-card-high {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #fee2e2;
    }
    .alert-card-medium {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.4);
        color: #fef3c7;
    }
    .alert-card-low {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #d1fae5;
    }
    .alert-card-clear {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.07);
        color: #94a3b8;
    }
    
    .alert-pill {
        display: inline-block;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 3px 8px;
        border-radius: 6px;
        margin-bottom: 6px;
    }
    .pill-high { background: #ef4444; color: #ffffff; }
    .pill-med { background: #f59e0b; color: #ffffff; }
    .pill-low { background: #10b981; color: #ffffff; }
    .pill-clear { background: #475569; color: #f1f5f9; }

    .alert-headline {
        font-size: 1.15rem;
        font-weight: 600;
        margin: 2px 0 4px 0;
        color: #f8fafc;
    }
    .alert-subtext {
        font-size: 0.82rem;
        opacity: 0.85;
    }

    /* Minimal Lane Corridor Visualizer */
    .corridor-bar {
        display: grid;
        grid-template-columns: 1fr 1.3fr 1fr;
        gap: 8px;
        margin-bottom: 14px;
    }
    .lane-cell {
        background: #1e293b;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 9px 8px;
        text-align: center;
        transition: all 0.2s ease;
    }
    .lane-name {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    .lane-status {
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 2px;
    }
    .lane-safe { color: #64748b; }
    .lane-warn-med { background: rgba(245, 158, 11, 0.15); border-color: #f59e0b; }
    .lane-warn-med .lane-name { color: #fde68a; }
    .lane-warn-med .lane-status { color: #f59e0b; }
    .lane-warn-high { background: rgba(239, 68, 68, 0.18); border-color: #ef4444; }
    .lane-warn-high .lane-name { color: #fca5a5; }
    .lane-warn-high .lane-status { color: #ef4444; }

    /* Minimal Telemetry KPI Cards */
    .telemetry-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-bottom: 14px;
    }
    .metric-mini-card {
        background: #172033;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 12px 14px;
    }
    .metric-mini-label {
        font-size: 0.72rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-mini-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 3px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .metric-mini-hint {
        font-size: 0.72rem;
        color: #64748b;
        margin-top: 2px;
    }

    /* Executive KPI Stats Ribbon for Analytics */
    .kpi-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 18px;
    }
    .analytics-kpi {
        background: #111827;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 18px;
    }
    .analytics-kpi-title {
        font-size: 0.75rem;
        font-weight: 500;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .analytics-kpi-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f8fafc;
        line-height: 1.2;
        margin: 4px 0 2px 0;
    }
    .analytics-kpi-sub {
        font-size: 0.74rem;
        color: #64748b;
    }

    /* Clean Streamlit tab bar polish */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        padding-bottom: 4px;
        margin-bottom: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 8px;
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 500;
        padding: 0 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(59, 130, 246, 0.12) !important;
        color: #60a5fa !important;
        font-weight: 600;
    }

    /* Compact DataFrame Styling */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Clean Video Controls & Prevent Text Wrapping */
    .stButton button, [data-testid="baseButton-secondary"], [data-testid="baseButton-primary"] {
        white-space: nowrap !important;
        word-break: keep-all !important;
        min-height: 38px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        font-weight: 500 !important;
    }

    .stSlider {
        margin-top: 6px !important;
        margin-bottom: 2px !important;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# AUTOMOTIVE SOUND ALERT ENGINE
# -----------------------------------------------------------------------------
# Powered by src.sound_alert (dual hardware speaker playback + browser HTML5 audio)


# -----------------------------------------------------------------------------
# SIDEBAR: MINIMAL SETTINGS & SYSTEM STATUS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Detection Settings")
    conf_day = st.slider("Daylight Sensitivity", 0.15, 0.80, float(CONFIDENCE_THRESHOLD), 0.05,
                         help="Minimum confidence threshold for daylight driving conditions.")
    conf_night = st.slider("Nighttime Sensitivity", 0.05, 0.40, float(CONFIDENCE_THRESHOLD_NIGHT), 0.01,
                           help="Permissive detection threshold for low-light/night footage.")
    
    st.markdown("---")
    st.markdown("### Sound Alerts")
    enable_audio = st.toggle("Enable Sound Alerts", value=True, help="Audible warning tones and voice driver alerts.")
    if enable_audio:
        audio_mode = st.selectbox(
            "Alert Audio Style",
            ["Chime + Spoken Voice", "Warning Chime Tone Only", "Spoken Voice Only"],
            index=0,
            help="Select alert style: automotive tones, natural voice speech, or both."
        )
        audio_vol = st.slider("Alert Volume", 20, 100, 80, 10, format="%d%%")
        
        c_sound_ph = st.empty()
        c_t1, c_t2 = st.columns(2)
        with c_t1:
            if st.button("🔊 Test Hazard", width='stretch', help="Test critical hazard alarm and voice"):
                play_sound_alert("High", "Warning, Deer ahead. Slow down.", mode=audio_mode, volume=audio_vol, force=True, container=c_sound_ph)
                st.toast("🔊 Playing Critical Hazard Alert...")
        with c_t2:
            if st.button("🔔 Test Caution", width='stretch', help="Test caution alert chime"):
                play_sound_alert("Medium", "Caution, animal on roadside.", mode=audio_mode, volume=audio_vol, force=True, container=c_sound_ph)
                st.toast("🔔 Playing Caution Alert...")
    else:
        audio_mode = "Chime + Spoken Voice"
        audio_vol = 80

    st.markdown("---")
    st.markdown("### Visual HUD")
    show_lanes = st.toggle("Show Corridor Lines", value=True, help="Draw left/center/right zone boundaries.")

    st.markdown("---")
    st.caption("**Model Engine**: YOLOv8s (Open Images V7)")
    st.caption("**Low-Light Enhancement**: Adaptive MSRCR + CLAHE")
    st.caption("**Road Zones**: Left (<35%) | Center | Right (>65%)")


# -----------------------------------------------------------------------------
# APP HEADER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="header-container">
    <div class="header-title-box">
        <h1><span>🛡️</span> SafeDrive ADAS</h1>
        <div class="header-subtitle">Animal Collision Warning & Driver Safety Intelligence</div>
    </div>
    <div class="status-badge">
        <span class="status-dot"></span> System Active
    </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_live, tab_analytics, tab_manager = st.tabs([
    "Live Detection & HUD",
    "BI Analytics & Insights",
    "Session Database & Tools"
])


# =============================================================================
# TAB 1: LIVE DETECTION & DRIVER HUD
# =============================================================================
with tab_live:
    # Top Control Bar (Clean Source & Mode Selector)
    data_dir = "data"
    sample_videos = glob.glob(os.path.join(data_dir, "*.mp4"))
    sample_names = [os.path.basename(p) for p in sample_videos] if sample_videos else []

    c_source, c_file, c_mode = st.columns([1.6, 2.4, 2])
    with c_source:
        source_mode = st.selectbox(
            "Source",
            ["Preloaded Clip", "Upload Video", "Live Camera"],
            label_visibility="collapsed"
        )
    
    selected_video_path = None
    camera_device_idx = 0
    camera_mode = "Live Stream"

    if source_mode == "Preloaded Clip":
        with c_file:
            preset = st.selectbox("Video Clip", sample_names if sample_names else ["No videos found"], label_visibility="collapsed")
            if sample_names:
                selected_video_path = os.path.join(data_dir, preset)
        with c_mode:
            playback_mode = st.selectbox(
                "Playback Mode",
                ["Interactive HUD Player", "Live YOLOv8 Inference"],
                label_visibility="collapsed",
                help="Interactive Player uses pre-indexed detections for fast playback. Live Inference evaluates each frame."
            )
    elif source_mode == "Upload Video":
        with c_file:
            uploaded_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"], label_visibility="collapsed")
            if uploaded_file:
                temp_upload_path = os.path.join("output", "uploaded_temp.mp4")
                os.makedirs("output", exist_ok=True)
                with open(temp_upload_path, "wb") as f:
                    f.write(uploaded_file.read())
                selected_video_path = temp_upload_path
        with c_mode:
            playback_mode = st.selectbox(
                "Playback Mode",
                ["Interactive HUD Player", "Live YOLOv8 Inference"],
                label_visibility="collapsed",
                help="Interactive Player uses pre-indexed detections for fast playback. Live Inference evaluates each frame."
            )
    else:  # Live Camera Mode
        with c_file:
            camera_device_idx = st.selectbox(
                "Camera Device",
                [0, 1, 2],
                format_func=lambda x: f"Webcam Device {x}",
                label_visibility="collapsed",
                help="Select your webcam device index (default is 0)."
            )
        with c_mode:
            camera_mode = st.selectbox(
                "Camera Mode",
                ["Live Stream", "Snapshot Photo"],
                label_visibility="collapsed",
                help="Live Stream processes continuous video feed. Snapshot Photo analyzes a single frame."
            )

    # Main Grid: Left Column Video / Camera, Right Column Driver HUD
    col_video, col_hud = st.columns([6.8, 5.2], gap="medium")

    # Shared Helper: Live Detector
    def run_detector_on_frame(frame_bgr: np.ndarray) -> list:
        from src.detector import AnimalDetector
        if "detector_instance" not in st.session_state:
            with st.spinner("Loading YOLOv8s model weights..."):
                st.session_state.detector_instance = AnimalDetector(
                    confidence=conf_day,
                    night_confidence=conf_night
                )
        
        detector = st.session_state.detector_instance
        detector.confidence = conf_day
        detector.night_confidence = conf_night
        raw_dets = detector.detect(frame_bgr, apply_preprocessing=True)
        for d in raw_dets:
            r, a = evaluate_detection_risk(d)
            d["risk"] = r
            d["alert"] = a
        return raw_dets

    # Placeholders in Right Column (HUD)
    with col_hud:
        alert_placeholder = st.empty()
        corridor_placeholder = st.empty()
        telemetry_placeholder = st.empty()
        table_placeholder = st.empty()

    # Shared Helper: Update HUD Display
    def update_hud_display(dets: list, is_night: bool, brightness: float, audio_ph=None):
        highest_risk = "None"
        active_alert = "No animal hazard detected in driving lane"
        active_animal = None
        active_zone = None
        max_area = 0.0
        max_conf = 0.0

        if dets:
            risk_ranks = {"High": 3, "Medium": 2, "Low": 1}
            sorted_dets = sorted(dets, key=lambda d: (risk_ranks.get(d.get("risk", "Low"), 0), d.get("area_fraction", 0)), reverse=True)
            top_det = sorted_dets[0]
            highest_risk = top_det.get("risk", "Low")
            active_alert = top_det.get("alert", "")
            active_animal = top_det.get("class_name", "Animal")
            active_zone = top_det.get("zone", "CENTER")
            max_area = top_det.get("area_fraction", 0.0)
            max_conf = top_det.get("confidence", 0.0)

        # 1. Alert Card
        if highest_risk == "High":
            alert_placeholder.markdown(f"""
            <div class="alert-card alert-card-high">
                <span class="alert-pill pill-high">Critical Hazard</span>
                <div class="alert-headline">{active_alert}</div>
                <div class="alert-subtext">Apply brakes and prepare to stop. Hazard in immediate trajectory.</div>
            </div>
            """, unsafe_allow_html=True)
            if enable_audio:
                play_sound_alert("High", active_alert, mode=audio_mode, volume=audio_vol, container=audio_ph)
        elif highest_risk == "Medium":
            alert_placeholder.markdown(f"""
            <div class="alert-card alert-card-medium">
                <span class="alert-pill pill-med">Caution Advised</span>
                <div class="alert-headline">{active_alert}</div>
                <div class="alert-subtext">Ease throttle and maintain safe separation.</div>
            </div>
            """, unsafe_allow_html=True)
            if enable_audio:
                play_sound_alert("Medium", active_alert, mode=audio_mode, volume=audio_vol, container=audio_ph)
        elif highest_risk == "Low":
            alert_placeholder.markdown(f"""
            <div class="alert-card alert-card-low">
                <span class="alert-pill pill-low">Advisory</span>
                <div class="alert-headline">{active_alert}</div>
                <div class="alert-subtext">Animal spotted on shoulder at safe distance. Maintain vigilant speed.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            alert_placeholder.markdown("""
            <div class="alert-card alert-card-clear">
                <span class="alert-pill pill-clear">Corridor Clear</span>
                <div class="alert-headline">Driving Path Unobstructed</div>
                <div class="alert-subtext">No animal hazards detected within immediate road bounds.</div>
            </div>
            """, unsafe_allow_html=True)

        # 2. Road Corridor 3-Lane Display
        zones_present = {d.get("zone"): d.get("risk", "Low") for d in dets}
        def zone_style(zone_name):
            if zone_name in zones_present:
                r = zones_present[zone_name]
                if r == "High":
                    return "lane-warn-high", "HAZARD"
                elif r == "Medium":
                    return "lane-warn-med", "CAUTION"
                return "lane-warn-med", "ACTIVE"
            return "", "CLEAR"

        l_class, l_status = zone_style("LEFT")
        c_class, c_status = zone_style("CENTER")
        r_class, r_status = zone_style("RIGHT")

        corridor_placeholder.markdown(f"""
        <div class="corridor-bar">
            <div class="lane-cell {l_class}">
                <div class="lane-name">Left Shoulder</div>
                <div class="lane-status {'lane-safe' if l_status=='CLEAR' else ''}">{l_status}</div>
            </div>
            <div class="lane-cell {c_class}">
                <div class="lane-name">Driving Lane</div>
                <div class="lane-status {'lane-safe' if c_status=='CLEAR' else ''}">{c_status}</div>
            </div>
            <div class="lane-cell {r_class}">
                <div class="lane-name">Right Shoulder</div>
                <div class="lane-status {'lane-safe' if r_status=='CLEAR' else ''}">{r_status}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 3. Clean Telemetry Grid
        env_mode = "Night Vision" if is_night else "Daylight"
        target_label = f"{active_animal} ({max_conf*100:.0f}%)" if active_animal else "None"
        prox_label = f"{max_area*100:.1f}% frame" if dets else "Clear"

        telemetry_placeholder.markdown(f"""
        <div class="telemetry-grid">
            <div class="metric-mini-card">
                <div class="metric-mini-label">Target Detected</div>
                <div class="metric-mini-value" style="color: #60a5fa;">{target_label}</div>
                <div class="metric-mini-hint">Zone: {active_zone or 'None'}</div>
            </div>
            <div class="metric-mini-card">
                <div class="metric-mini-label">Proximity Proxy</div>
                <div class="metric-mini-value" style="color: #f59e0b;">{prox_label}</div>
                <div class="metric-mini-hint">BBox area fraction</div>
            </div>
            <div class="metric-mini-card">
                <div class="metric-mini-label">Lighting Mode</div>
                <div class="metric-mini-value" style="color: #a78bfa;">{env_mode}</div>
                <div class="metric-mini-hint">Brightness: {brightness:.1f} / 255</div>
            </div>
            <div class="metric-mini-card">
                <div class="metric-mini-label">Animals in View</div>
                <div class="metric-mini-value" style="color: #34d399;">{len(dets)}</div>
                <div class="metric-mini-hint">Active tracking tracks</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 4. Clean Detections Table
        if dets:
            df_curr = pd.DataFrame(dets)[["class_name", "confidence", "zone", "risk"]]
            df_curr.columns = ["Animal", "Confidence", "Lane", "Risk Level"]
            df_curr["Confidence"] = df_curr["Confidence"].map(lambda x: f"{x*100:.1f}%")
            table_placeholder.dataframe(df_curr, width='stretch', hide_index=True)
        else:
            table_placeholder.caption("No animal detection in active frame.")


    # -------------------------------------------------------------------------
    # CASE A: LIVE WEBCAM / CAMERA MODE
    # -------------------------------------------------------------------------
    if source_mode == "Live Camera":
        with col_video:
            video_placeholder = st.empty()
            audio_placeholder = st.empty()
            status_line = st.empty()

            if camera_mode == "Live Stream":
                c_cam_ctrl1, c_cam_ctrl2 = st.columns([2.5, 2], gap="small")
                with c_cam_ctrl1:
                    cam_active = st.toggle("Activate Live Camera Feed", value=False, key="live_webcam_stream_active")
                with c_cam_ctrl2:
                    proc_throttle = st.selectbox(
                        "Inference Rate",
                        ["Process Every Frame", "Every 2nd Frame (Fast)"],
                        label_visibility="collapsed"
                    )

                if cam_active:
                    cap_live = cv2.VideoCapture(camera_device_idx)
                    cap_live.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap_live.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                    if not cap_live.isOpened():
                        st.error(f"Could not open Camera Device {camera_device_idx}. Please check connection or device permissions.")
                    else:
                        frame_count = 0
                        t_start = time.time()
                        last_dets = []
                        skip_rate = 2 if "Fast" in proc_throttle else 1

                        try:
                            while st.session_state.get("live_webcam_stream_active", False):
                                ret, frame = cap_live.read()
                                if not ret or frame is None:
                                    st.warning("Camera stream interrupted.")
                                    break

                                frame_count += 1

                                # Run detection with rate control
                                if frame_count % skip_rate == 0 or not last_dets:
                                    last_dets = run_detector_on_frame(frame)

                                # Draw bounding boxes
                                annotated_bgr = draw_detections(
                                    frame.copy(),
                                    last_dets,
                                    draw_lanes=show_lanes,
                                    left_zone_end=LEFT_ZONE_END,
                                    right_zone_start=RIGHT_ZONE_START
                                )
                                annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                                video_placeholder.image(annotated_rgb, width='stretch')

                                # Analyze lighting
                                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                brightness = float(np.mean(gray))
                                is_night = brightness < VERY_DARK_BRIGHTNESS

                                # Update HUD
                                update_hud_display(last_dets, is_night, brightness, audio_ph=audio_placeholder)

                                # Live telemetry badge
                                elapsed = time.time() - t_start
                                current_fps = frame_count / elapsed if elapsed > 0 else 0.0
                                status_line.markdown(
                                    f"<div style='font-size: 0.78rem; color: #10b981; margin-top: -6px; text-align: right;'>"
                                    f"● Live Camera Streaming &nbsp;|&nbsp; <b>{current_fps:.1f} FPS</b> &nbsp;|&nbsp; Frame #{frame_count}"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
                                time.sleep(0.015)
                        finally:
                            cap_live.release()
                else:
                    # Camera standby view
                    video_placeholder.markdown("""
                    <div class="clean-card" style="text-align: center; padding: 75px 20px; border: 1px dashed rgba(255,255,255,0.14);">
                        <div style="font-size: 2.8rem; margin-bottom: 12px;">📹</div>
                        <div style="font-size: 1.15rem; font-weight: 600; color: #f8fafc;">Live Camera Standby</div>
                        <div style="font-size: 0.85rem; color: #94a3b8; max-width: 420px; margin: 8px auto 0 auto; line-height: 1.4;">
                            Toggle <b>Activate Live Camera Feed</b> above to begin real-time animal detection and risk assessment from your webcam.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    update_hud_display([], is_night=False, brightness=120.0, audio_ph=None)

            else:  # Snapshot Photo Mode
                camera_photo = st.camera_input("Capture road / animal snapshot from camera", label_visibility="collapsed")
                if camera_photo is not None:
                    bytes_data = camera_photo.getvalue()
                    img_bgr = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    
                    if img_bgr is not None:
                        dets = run_detector_on_frame(img_bgr)
                        annotated_bgr = draw_detections(
                            img_bgr.copy(),
                            dets,
                            draw_lanes=show_lanes,
                            left_zone_end=LEFT_ZONE_END,
                            right_zone_start=RIGHT_ZONE_START
                        )
                        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(annotated_rgb, width='stretch')

                        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                        brightness = float(np.mean(gray))
                        is_night = brightness < VERY_DARK_BRIGHTNESS

                        update_hud_display(dets, is_night, brightness, audio_ph=audio_placeholder)
                        status_line.caption(f"Snapshot analyzed: {len(dets)} animal detections identified.")
                else:
                    video_placeholder.info("Take a photo above to run instant animal detection & corridor risk assessment.")
                    update_hud_display([], is_night=False, brightness=120.0, audio_ph=None)


    # -------------------------------------------------------------------------
    # CASE B: VIDEO CLIP PLAYBACK (PRELOADED OR UPLOADED)
    # -------------------------------------------------------------------------
    else:
        if not selected_video_path or not os.path.exists(selected_video_path):
            st.info("Please select or upload a video clip to begin detection monitoring.")
            st.stop()

        # Load video properties
        cap = cv2.VideoCapture(selected_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration_sec = total_frames / fps
        cap.release()

        # Load cached detections if available
        cached_detections = {}
        base_name = os.path.basename(selected_video_path)
        base_stem = os.path.splitext(base_name)[0]
        specific_json = f"output/{base_stem}_detections.json"
        default_json = "output/detections.json"

        json_to_load = None
        if os.path.exists(specific_json):
            json_to_load = specific_json
        elif "cep.mp4" in base_name and os.path.exists(default_json):
            json_to_load = default_json

        if json_to_load:
            try:
                with open(json_to_load, "r") as f:
                    raw_dets = json.load(f)
                    for item in raw_dets:
                        cached_detections[item["frame_index"]] = item.get("detections", [])
            except Exception:
                cached_detections = {}

        # Playback Session State
        if "is_playing" not in st.session_state:
            st.session_state.is_playing = False
        if "current_frame" not in st.session_state:
            st.session_state.current_frame = 0

        # Helper: Get frame detections
        def get_detections_for_frame(frame_idx: int, frame_bgr: np.ndarray, run_live: bool = False):
            if not run_live and frame_idx in cached_detections:
                dets = [dict(d) for d in cached_detections[frame_idx]]
                for d in dets:
                    r, a = evaluate_detection_risk(d)
                    d["risk"] = r
                    d["alert"] = a
                return dets

            return run_detector_on_frame(frame_bgr)

        # Left Column: Video Feed & Clean Media Bar
        with col_video:
            video_placeholder = st.empty()
            audio_placeholder = st.empty()

            # 1. Full-Width Timeline Scrubber directly beneath video
            scrub_frame = st.slider(
                "Timeline Scrubber",
                min_value=0,
                max_value=max(1, total_frames - 1),
                value=st.session_state.current_frame,
                label_visibility="collapsed"
            )
            if scrub_frame != st.session_state.current_frame:
                st.session_state.current_frame = scrub_frame

            # 2. Balanced Media Control Bar (Never squished or wrapped)
            c_play, c_rst, c_time, c_spd = st.columns([1.5, 1.5, 4.5, 2.2], gap="small")
            
            with c_play:
                play_label = "⏸ Pause" if st.session_state.is_playing else "▶ Play"
                if st.button(play_label, width='stretch'):
                    st.session_state.is_playing = not st.session_state.is_playing
            
            with c_rst:
                if st.button("↺ Reset", width='stretch'):
                    st.session_state.current_frame = 0
                    st.session_state.is_playing = False

            with c_time:
                curr_time = st.session_state.current_frame / fps
                st.markdown(
                    f"<div style='text-align: center; padding-top: 8px; font-size: 0.85rem; color: #94a3b8; font-weight: 500; white-space: nowrap;'>"
                    f"⏱ <b>{curr_time:.2f}s</b> / {duration_sec:.2f}s &nbsp;•&nbsp; Frame <b>{st.session_state.current_frame}</b> of {total_frames}"
                    f"</div>",
                    unsafe_allow_html=True
                )

            with c_spd:
                playback_speed = st.selectbox(
                    "Speed",
                    [1.0, 1.5, 2.0],
                    index=0,
                    format_func=lambda x: f"{x}x Speed",
                    label_visibility="collapsed"
                )

        # Render frame function
        def render_hud_state(frame_idx: int):
            cap_seek = cv2.VideoCapture(selected_video_path)
            cap_seek.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap_seek.read()
            cap_seek.release()

            if not ret or frame is None:
                return None, []

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            is_night = brightness < VERY_DARK_BRIGHTNESS

            is_live = (playback_mode == "Live YOLOv8 Inference")
            dets = get_detections_for_frame(frame_idx, frame, run_live=is_live)

            # Draw bounding boxes & zones
            annotated_bgr = frame.copy()
            annotated_bgr = draw_detections(
                annotated_bgr,
                dets,
                draw_lanes=show_lanes,
                left_zone_end=LEFT_ZONE_END,
                right_zone_start=RIGHT_ZONE_START
            )
            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

            # Update HUD
            update_hud_display(dets, is_night, brightness, audio_ph=audio_placeholder)

            return annotated_rgb, dets

        # First Frame Rendering
        current_img, frame_dets = render_hud_state(st.session_state.current_frame)
        if current_img is not None:
            video_placeholder.image(current_img, width='stretch')

        # Active Playback Loop
        if st.session_state.is_playing:
            target_fps = fps * playback_speed
            frame_delay = max(0.015, 1.0 / target_fps)
            step = 1 if playback_speed <= 1.0 else 2

            while st.session_state.is_playing and st.session_state.current_frame < total_frames - 1:
                st.session_state.current_frame = min(total_frames - 1, st.session_state.current_frame + step)
                img_step, _ = render_hud_state(st.session_state.current_frame)
                if img_step is not None:
                    video_placeholder.image(img_step, width='stretch')
                time.sleep(frame_delay)
                if st.session_state.current_frame >= total_frames - 1:
                    st.session_state.is_playing = False
                    st.rerun()


# =============================================================================
# TAB 2: BI ANALYTICS & INSIGHTS
# =============================================================================
with tab_analytics:
    sessions = load_all_sessions()
    events_df = get_all_events_df(sessions)
    summary_df = get_sessions_summary_df(sessions)

    # 1. Clean KPI Ribbon
    tot_sessions = len(summary_df)
    tot_incidents = len(events_df)
    tot_high = len(events_df[events_df["risk"] == "High"])
    night_incidents = len(events_df[events_df["lighting"] == "Night"])
    night_pct = (night_incidents / tot_incidents * 100) if tot_incidents > 0 else 0.0

    st.markdown(f"""
    <div class="kpi-row">
        <div class="analytics-kpi">
            <div class="analytics-kpi-title">Monitored Drives</div>
            <div class="analytics-kpi-val">{tot_sessions}</div>
            <div class="analytics-kpi-sub">Registered sessions</div>
        </div>
        <div class="analytics-kpi">
            <div class="analytics-kpi-title">Animal Incidents</div>
            <div class="analytics-kpi-val">{tot_incidents}</div>
            <div class="analytics-kpi-sub">Total encounters tracked</div>
        </div>
        <div class="analytics-kpi">
            <div class="analytics-kpi-title">High-Risk Threats</div>
            <div class="analytics-kpi-val" style="color: #f87171;">{tot_high}</div>
            <div class="analytics-kpi-sub">Critical collision alerts</div>
        </div>
        <div class="analytics-kpi">
            <div class="analytics-kpi-title">Night Incident Share</div>
            <div class="analytics-kpi-val" style="color: #a78bfa;">{night_pct:.1f}%</div>
            <div class="analytics-kpi-sub">{night_incidents} low-light encounters</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 2. Compact Filter Strip
    c_f1, c_f2, c_f3 = st.columns([2, 1.5, 1.5])
    with c_f1:
        s_opts = ["All Sessions"] + list(summary_df["session_name"].unique())
        f_session = st.selectbox("Session Filter", s_opts, label_visibility="collapsed")
    with c_f2:
        f_lighting = st.selectbox("Lighting", ["All Conditions", "Day", "Night"], label_visibility="collapsed")
    with c_f3:
        f_risk = st.selectbox("Risk Level", ["All Severities", "High", "Medium", "Low"], label_visibility="collapsed")

    # Apply filters
    filtered_df = events_df.copy()
    if f_session != "All Sessions":
        filtered_df = filtered_df[filtered_df["session_name"] == f_session]
    if f_lighting != "All Conditions":
        filtered_df = filtered_df[filtered_df["lighting"] == f_lighting]
    if f_risk != "All Severities":
        filtered_df = filtered_df[filtered_df["risk"] == f_risk]

    # Clean Chart Theming
    plotly_layout_defaults = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", family="Inter, sans-serif"),
        margin=dict(l=20, r=20, t=35, b=30),
    )

    # Clean Sub-Tabs for Analytics
    sub_trends, sub_risk, sub_night, sub_export = st.tabs([
        "Overview & Trends",
        "Threat Breakdown",
        "Day vs. Night Analysis",
        "Incident Log & Export"
    ])

    with sub_trends:
        c_tr1, c_tr2 = st.columns(2)
        with c_tr1:
            fig_sess = go.Figure()
            fig_sess.add_trace(go.Bar(
                x=summary_df["session_name"],
                y=summary_df["total_events"],
                name="Total Encounters",
                marker_color="#3b82f6",
                text=summary_df["total_events"],
                textposition="auto"
            ))
            fig_sess.add_trace(go.Bar(
                x=summary_df["session_name"],
                y=summary_df["high_risk"],
                name="High Risk",
                marker_color="#ef4444",
                text=summary_df["high_risk"],
                textposition="auto"
            ))
            fig_sess.update_layout(
                **plotly_layout_defaults,
                title="Incidents & High-Risk Alerts by Session",
                barmode="group",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_sess, width='stretch')

        with c_tr2:
            fig_scat = px.scatter(
                filtered_df,
                x="start_time",
                y="animal",
                color="risk",
                size="duration_sec",
                color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
                hover_data=["session_name", "position", "confidence"],
                title="Incident Timeline (Drive Seconds vs. Species Encountered)"
            )
            fig_scat.update_layout(
                **plotly_layout_defaults,
                xaxis_title="Drive Timestamp (s)",
                yaxis_title="Species"
            )
            st.plotly_chart(fig_scat, width='stretch')

    with sub_risk:
        c_rk1, c_rk2 = st.columns([1, 1.3])
        with c_rk1:
            risk_counts = filtered_df["risk"].value_counts().reset_index()
            risk_counts.columns = ["Risk", "Count"]
            fig_donut = px.pie(
                risk_counts,
                names="Risk",
                values="Count",
                hole=0.6,
                color="Risk",
                color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
                title="Threat Severity Distribution"
            )
            fig_donut.update_layout(**plotly_layout_defaults)
            st.plotly_chart(fig_donut, width='stretch')

        with c_rk2:
            fig_species = px.histogram(
                filtered_df,
                x="animal",
                color="risk",
                barmode="stack",
                color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
                title="Species Risk Profile"
            )
            fig_species.update_layout(
                **plotly_layout_defaults,
                xaxis_title="Species",
                yaxis_title="Encounters"
            )
            st.plotly_chart(fig_species, width='stretch')

    with sub_night:
        day_events = events_df[events_df["lighting"] == "Day"]
        night_events = events_df[events_df["lighting"] == "Night"]

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Daylight Incidents", f"{len(day_events)} Encounters")
        m2.metric("Nighttime Incidents", f"{len(night_events)} Encounters")
        
        day_h_rate = (len(day_events[day_events["risk"]=="High"])/len(day_events)*100) if len(day_events)>0 else 0
        night_h_rate = (len(night_events[night_events["risk"]=="High"])/len(night_events)*100) if len(night_events)>0 else 0
        m3.metric("Night High-Risk Rate", f"{night_h_rate:.1f}%", delta=f"{night_h_rate - day_h_rate:+.1f}% vs Day")
        
        day_c = day_events["confidence"].mean() if not day_events.empty else 0
        night_c = night_events["confidence"].mean() if not night_events.empty else 0
        m4.metric("Night Avg Confidence", f"{night_c*100:.1f}%", delta=f"{(night_c - day_c)*100:+.1f}% vs Day")

        c_dn1, c_dn2 = st.columns(2)
        with c_dn1:
            fig_dn_s = px.histogram(
                events_df,
                x="animal",
                color="lighting",
                barmode="group",
                color_discrete_map={"Day": "#f59e0b", "Night": "#8b5cf6"},
                title="Species Observed: Day vs. Night"
            )
            fig_dn_s.update_layout(**plotly_layout_defaults, xaxis_title="Species", yaxis_title="Count")
            st.plotly_chart(fig_dn_s, width='stretch')

        with c_dn2:
            fig_dn_r = px.histogram(
                events_df,
                x="lighting",
                color="risk",
                barmode="stack",
                color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
                title="Severity Proportions by Lighting"
            )
            fig_dn_r.update_layout(**plotly_layout_defaults, xaxis_title="Lighting Condition", yaxis_title="Incidents")
            st.plotly_chart(fig_dn_r, width='stretch')

    with sub_export:
        export_cols = ["session_name", "lighting", "animal", "risk", "position", "start_time", "duration_sec", "confidence", "alert"]
        clean_exp = filtered_df[export_cols].copy()
        clean_exp.columns = ["Session", "Lighting", "Animal", "Risk", "Lane", "Time (s)", "Duration (s)", "Confidence", "Alert Message"]
        st.dataframe(clean_exp, width='stretch', hide_index=True)

        d1, d2, _ = st.columns([1.5, 1.5, 4])
        with d1:
            st.download_button(
                "Export CSV",
                clean_exp.to_csv(index=False).encode("utf-8"),
                file_name="animal_incidents.csv",
                mime="text/csv",
                width='stretch'
            )
        with d2:
            st.download_button(
                "Export JSON",
                json.dumps(json.loads(clean_exp.to_json(orient="records")), indent=2),
                file_name="animal_incidents.json",
                mime="application/json",
                width='stretch'
            )


# =============================================================================
# TAB 3: SESSION DATABASE & TOOLS
# =============================================================================
with tab_manager:
    st.markdown("#### Registered Driving Sessions")
    sessions = load_all_sessions()
    summary_df = get_sessions_summary_df(sessions)
    st.dataframe(summary_df, width='stretch', hide_index=True)

    st.markdown("---")
    st.markdown("#### Analyze New Video Clip")
    with st.container():
        all_vids = glob.glob("data/*.mp4")
        c_p1, c_p2, c_p3 = st.columns([2, 2, 1.5])
        with c_p1:
            v_proc = st.selectbox("Video File", all_vids)
        with c_p2:
            s_name = st.text_input("Session Label", value=f"Drive: {os.path.basename(v_proc) if v_proc else 'Clip'}")
        with c_p3:
            s_light = st.selectbox("Lighting", ["Night", "Day", "Twilight"])

        if st.button("Run Full Detection Pipeline", type="primary"):
            with st.status("Running ADAS detection & tracking pipeline...", expanded=True) as status:
                st.write("Initializing detector and reading frames...")
                from main import run_on_video
                out_vid = f"output/{os.path.basename(v_proc)}"
                out_json = f"output/{os.path.splitext(os.path.basename(v_proc))[0]}_detections.json"

                frames_log, events_log = run_on_video(v_proc, out_vid, out_json)
                st.write(f"Completed: {len(frames_log)} frames analyzed, {len(events_log)} distinct animal events cataloged.")

                cap_b = cv2.VideoCapture(v_proc)
                ret_b, f_b = cap_b.read()
                cap_b.release()
                mb = float(np.mean(cv2.cvtColor(f_b, cv2.COLOR_BGR2GRAY))) if ret_b else 50.0

                new_s = compute_session_metrics(
                    session_id=f"session_{int(time.time())}",
                    session_name=s_name,
                    video_name=os.path.basename(v_proc),
                    lighting=s_light,
                    mean_brightness=mb,
                    fps=30.0,
                    frame_logs=frames_log,
                    events=events_log
                )
                add_or_update_session(new_s)
                status.update(label="Detection pipeline complete & synced with database!", state="complete")
                st.success(f"Session **{s_name}** added to analytics.")
                st.rerun()
