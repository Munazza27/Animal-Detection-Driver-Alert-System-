"""
app.py
------
Animal Detection & Driver Alert System (ADAS)
- Live Demo Interface: Video playback with bounding box overlay + color-coded risk indicator
- Simple BI-Style Analytics Panel: Detections per session, risk-level distribution, Day vs. Night comparison
"""

import os
import time
import json
import glob
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
    add_or_update_session
)

# Page configuration
st.set_page_config(
    page_title="Animal Detection & Driver Alert System",
    page_icon="🦌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom HUD & Dashboard CSS Styling
st.markdown("""
<style>
    /* Global polish */
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }
    
    /* Modern Card Container */
    .hud-card {
        background: #111827;
        border-radius: 12px;
        padding: 16px 20px;
        border: 1px solid #1f2937;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    /* Driver HUD Risk Banners */
    .risk-banner-high {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.25), rgba(185, 28, 28, 0.45));
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 16px 20px;
        color: #fecaca;
        text-align: center;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
        animation: pulse-danger 1.5s infinite alternate;
    }
    
    .risk-banner-medium {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.22), rgba(180, 83, 9, 0.4));
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 16px 20px;
        color: #fef3c7;
        text-align: center;
        box-shadow: 0 0 15px rgba(245, 158, 11, 0.3);
    }
    
    .risk-banner-low {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.18), rgba(5, 150, 105, 0.3));
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 16px 20px;
        color: #d1fae5;
        text-align: center;
    }
    
    .risk-banner-clear {
        background: linear-gradient(135deg, rgba(55, 65, 81, 0.4), rgba(31, 41, 55, 0.6));
        border: 1px solid #4b5563;
        border-radius: 12px;
        padding: 16px 20px;
        color: #9ca3af;
        text-align: center;
    }
    
    /* Road Corridor Zones */
    .zone-box {
        padding: 10px;
        text-align: center;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .zone-inactive {
        background: #1f2937;
        color: #6b7280;
        border: 1px solid #374151;
    }
    .zone-active-high {
        background: rgba(239, 68, 68, 0.3);
        color: #fca5a5;
        border: 2px solid #ef4444;
        box-shadow: 0 0 10px rgba(239, 68, 68, 0.5);
    }
    .zone-active-med {
        background: rgba(245, 158, 11, 0.3);
        color: #fde68a;
        border: 2px solid #f59e0b;
    }
    .zone-active-low {
        background: rgba(16, 185, 129, 0.3);
        color: #a7f3d0;
        border: 2px solid #10b981;
    }

    /* KPI Stat Cards */
    .kpi-card {
        background: #1f2937;
        border-radius: 10px;
        padding: 14px 18px;
        border-left: 4px solid #3b82f6;
    }
    .kpi-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #9ca3af;
        margin-bottom: 4px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f9fafb;
        line-height: 1.2;
    }
    .kpi-subtitle {
        font-size: 0.75rem;
        color: #6b7280;
        margin-top: 4px;
    }

    @keyframes pulse-danger {
        0% { box-shadow: 0 0 10px rgba(239, 68, 68, 0.3); }
        100% { box-shadow: 0 0 25px rgba(239, 68, 68, 0.7); }
    }
</style>
""", unsafe_allow_html=True)


# Audio alert helper (Web Audio API synthesis for seamless client-side audio)
def play_audio_alert_script(risk: str):
    """Generates an embedded JavaScript Web Audio beep based on risk level."""
    if risk == "High":
        freq1, freq2 = 880, 1174  # High urgent alarm tone
        dur = 0.3
    elif risk == "Medium":
        freq1, freq2 = 587, 784   # Caution alert chime
        dur = 0.25
    else:
        return ""

    js_code = f"""
    <script>
    (function() {{
        try {{
            var AudioContext = window.AudioContext || window.webkitAudioContext;
            if (!AudioContext) return;
            var ctx = new AudioContext();
            var osc = ctx.createOscillator();
            var gain = ctx.createGain();
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.type = 'triangle';
            osc.frequency.setValueAtTime({freq1}, ctx.currentTime);
            osc.frequency.exponentialRampToValueAtTime({freq2}, ctx.currentTime + {dur});
            gain.gain.setValueAtTime(0.3, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + {dur});
            osc.start();
            osc.stop(ctx.currentTime + {dur});
        }} catch(e) {{}}
    }})();
    </script>
    """
    return js_code


# -------------------------------------------------------------
# SIDEBAR CONFIGURATION
# -------------------------------------------------------------
st.sidebar.markdown("### 🦌 ADAS System Controls")
nav_choice = st.sidebar.radio(
    "Navigation",
    ["🎥 Live Demo & Driver HUD", "📊 BI Analytics Panel", "⚙️ Session Manager & Logs"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("#### ⚙️ Detection & Alert Settings")
conf_day = st.sidebar.slider("Daylight Confidence", 0.15, 0.80, float(CONFIDENCE_THRESHOLD), 0.05)
conf_night = st.sidebar.slider("Nighttime Confidence", 0.05, 0.40, float(CONFIDENCE_THRESHOLD_NIGHT), 0.01)
enable_audio = st.sidebar.checkbox("🔊 Enable Driver Audio Alerts", value=True)
show_lanes = st.sidebar.checkbox("🛣️ Show Corridor Guide Lines", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("#### 💡 System Telemetry")
st.sidebar.info("""
**Model**: YOLOv8s (Open Images V7)  
**Classes**: 15 Road Animal Species  
**Low-Light Engine**: Adaptive MSRCR + CLAHE  
**Zones**: Left (<35%) | Center | Right (>65%)
""")


# =============================================================
# TAB 1: LIVE DEMO INTERFACE & DRIVER HUD
# =============================================================
if nav_choice == "🎥 Live Demo & Driver HUD":
    st.title("🎥 Live Demo Interface & Driver Alert HUD")
    st.markdown("Real-time video playback with bounding box overlay, road corridor tracking, and color-coded risk indicator.")

    # 1. Video Selection
    data_dir = "data"
    sample_videos = glob.glob(os.path.join(data_dir, "*.mp4"))
    sample_names = [os.path.basename(p) for p in sample_videos] if sample_videos else []

    col_v1, col_v2, col_v3 = st.columns([2, 2, 2])
    with col_v1:
        video_source_mode = st.selectbox("Video Source", ["Preloaded Test Clip", "Upload Custom Video"])
    
    selected_video_path = None
    if video_source_mode == "Preloaded Test Clip":
        with col_v2:
            preset = st.selectbox(
                "Select Clip",
                sample_names if sample_names else ["No sample videos found"],
                index=0 if sample_names else 0
            )
            if sample_names:
                selected_video_path = os.path.join(data_dir, preset)
    else:
        with col_v2:
            uploaded_file = st.file_uploader("Upload MP4 / MOV Video", type=["mp4", "mov", "avi"])
            if uploaded_file:
                temp_upload_path = os.path.join("output", "uploaded_temp.mp4")
                os.makedirs("output", exist_ok=True)
                with open(temp_upload_path, "wb") as f:
                    f.write(uploaded_file.read())
                selected_video_path = temp_upload_path

    if not selected_video_path or not os.path.exists(selected_video_path):
        st.warning("Please select or upload a valid video to begin.")
        st.stop()

    # Load video metadata
    cap = cv2.VideoCapture(selected_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration_sec = total_frames / fps
    w_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Load pre-computed detections if this is cep.mp4 or has matching json
    cached_detections = {}
    base_name = os.path.basename(selected_video_path)
    if "cep.mp4" in base_name and os.path.exists("output/detections.json"):
        with open("output/detections.json", "r") as f:
            raw_dets = json.load(f)
            for item in raw_dets:
                cached_detections[item["frame_index"]] = item.get("detections", [])

    with col_v3:
        playback_mode = st.selectbox(
            "Playback Mode",
            ["⚡ Interactive HUD Player", "🤖 Live YOLOv8 Inference"],
            help="Interactive HUD Player gives high-FPS smooth playback with scrub control. Live Inference runs the model frame-by-frame on CPU."
        )

    st.markdown("---")

    # Layout: Left column for Video & Controls, Right column for Driver HUD & Telemetry
    col_vid, col_hud = st.columns([7, 5])

    # Session state for playback
    if "is_playing" not in st.session_state:
        st.session_state.is_playing = False
    if "current_frame" not in st.session_state:
        st.session_state.current_frame = 0

    with col_vid:
        st.markdown("##### 🚗 Dashcam Road Feed")
        video_placeholder = st.empty()
        audio_placeholder = st.empty()

        # Playback control bar
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.5, 1.5, 3, 2])
        with ctrl_col1:
            if st.button("▶ Play" if not st.session_state.is_playing else "⏸ Pause", use_container_width=True):
                st.session_state.is_playing = not st.session_state.is_playing
        with ctrl_col2:
            if st.button("🔄 Restart", use_container_width=True):
                st.session_state.current_frame = 0
                st.session_state.is_playing = False
        with ctrl_col3:
            playback_speed = st.select_slider(
                "Playback Speed",
                options=[0.5, 1.0, 1.5, 2.0],
                value=1.0,
                format_func=lambda x: f"{x}x"
            )
        with ctrl_col4:
            frame_stride = st.selectbox("Frame Step", [1, 2, 3, 5], index=0, help="Skip frames for faster playback")

        # Frame seeker slider
        slider_frame = st.slider(
            "Timeline Scrubber (Frame & Timestamp)",
            min_value=0,
            max_value=max(1, total_frames - 1),
            value=st.session_state.current_frame,
            format="%d frame"
        )
        if slider_frame != st.session_state.current_frame:
            st.session_state.current_frame = slider_frame

        current_sec = st.session_state.current_frame / fps
        st.caption(f"⏱ Current Time: **{current_sec:.2f}s** / {duration_sec:.2f}s | Frame: **{st.session_state.current_frame}** of {total_frames}")

    # Telemetry HUD in right column
    with col_hud:
        st.markdown("##### 🛡️ Driver Alert HUD & Risk Status")
        risk_banner_placeholder = st.empty()
        corridor_placeholder = st.empty()
        metrics_placeholder = st.empty()
        alert_log_placeholder = st.empty()

    # Function to get frame detections
    def get_detections_for_frame(frame_idx: int, frame_bgr: np.ndarray, run_live: bool = False):
        if not run_live and frame_idx in cached_detections:
            dets = [dict(d) for d in cached_detections[frame_idx]]
            # Attach risk and alert info
            for d in dets:
                r, a = evaluate_detection_risk(d)
                d["risk"] = r
                d["alert"] = a
            return dets

        # If live or not cached, run detector
        from src.detector import AnimalDetector
        if "detector_instance" not in st.session_state:
            with st.spinner("Initializing YOLOv8 model..."):
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

    # Render a single frame and update HUD
    def render_hud_state(frame_idx: int):
        cap_seek = cv2.VideoCapture(selected_video_path)
        cap_seek.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap_seek.read()
        cap_seek.release()

        if not ret or frame is None:
            return None, []

        # Analyze lighting
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray))
        is_night = brightness < VERY_DARK_BRIGHTNESS

        is_live = (playback_mode == "🤖 Live YOLOv8 Inference")
        dets = get_detections_for_frame(frame_idx, frame, run_live=is_live)

        # Draw overlays
        annotated_bgr = frame.copy()
        annotated_bgr = draw_detections(
            annotated_bgr,
            dets,
            draw_lanes=show_lanes,
            left_zone_end=LEFT_ZONE_END,
            right_zone_start=RIGHT_ZONE_START
        )
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        # Determine Highest Risk & Active Alerts
        highest_risk = "None"
        active_alert = None
        active_animal = None
        active_zone = None
        max_area = 0.0
        max_conf = 0.0

        if dets:
            risk_ranks = {"High": 3, "Medium": 2, "Low": 1}
            sorted_dets = sorted(dets, key=lambda d: (risk_ranks.get(d.get("risk", "Low"), 0), d.get("area_fraction", 0)), reverse=True)
            top_det = sorted_dets[0]
            highest_risk = top_det.get("risk", "Low")
            active_alert = top_det.get("alert")
            active_animal = top_det.get("class_name")
            active_zone = top_det.get("zone", "CENTER")
            max_area = top_det.get("area_fraction", 0.0)
            max_conf = top_det.get("confidence", 0.0)

        # Update Risk Banner
        if highest_risk == "High":
            risk_banner_placeholder.markdown(f"""
            <div class="risk-banner-high">
                <div style="font-size: 1.6rem; font-weight: 800; letter-spacing: 0.05em;">🔴 CRITICAL HAZARD — HIGH RISK</div>
                <div style="font-size: 1.15rem; font-weight: 600; margin-top: 6px;">{active_alert}</div>
                <div style="font-size: 0.85rem; margin-top: 4px; opacity: 0.9;">IMMEDIATE DRIVER ACTION: Apply Brakes & Sound Horn</div>
            </div>
            """, unsafe_allow_html=True)
            if enable_audio:
                audio_placeholder.markdown(play_audio_alert_script("High"), unsafe_allow_html=True)

        elif highest_risk == "Medium":
            risk_banner_placeholder.markdown(f"""
            <div class="risk-banner-medium">
                <div style="font-size: 1.5rem; font-weight: 800; letter-spacing: 0.05em;">🟡 CAUTION — MEDIUM RISK</div>
                <div style="font-size: 1.1rem; font-weight: 600; margin-top: 6px;">{active_alert}</div>
                <div style="font-size: 0.85rem; margin-top: 4px; opacity: 0.9;">ADVISORY: Release Throttle & Prepare to Yield</div>
            </div>
            """, unsafe_allow_html=True)
            if enable_audio:
                audio_placeholder.markdown(play_audio_alert_script("Medium"), unsafe_allow_html=True)

        elif highest_risk == "Low":
            risk_banner_placeholder.markdown(f"""
            <div class="risk-banner-low">
                <div style="font-size: 1.4rem; font-weight: 700;">🟢 ADVISORY — LOW RISK</div>
                <div style="font-size: 1.0rem; font-weight: 500; margin-top: 4px;">{active_alert}</div>
                <div style="font-size: 0.8rem; margin-top: 4px; opacity: 0.85;">Animal detected at safe distance. Maintain vigilant speed.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            risk_banner_placeholder.markdown("""
            <div class="risk-banner-clear">
                <div style="font-size: 1.3rem; font-weight: 700;">✅ ROAD CORRIDOR CLEAR</div>
                <div style="font-size: 0.95rem; margin-top: 4px;">No immediate animal collision hazard detected in driving lane.</div>
            </div>
            """, unsafe_allow_html=True)

        # Update 3-Zone Corridor HUD
        zones_present = {d.get("zone"): d.get("risk", "Low") for d in dets}
        def get_zone_class(zone_name):
            if zone_name in zones_present:
                r = zones_present[zone_name]
                return f"zone-active-{r.lower()[:3]}"
            return "zone-inactive"

        corridor_placeholder.markdown(f"""
        <div style="display: flex; gap: 8px; margin-top: 14px; margin-bottom: 14px;">
            <div class="zone-box {get_zone_class('LEFT')}" style="flex: 1;">
                LEFT ROADSIDE<br><span style="font-size: 0.75rem;">{'⚠️ ACTIVE' if 'LEFT' in zones_present else 'CLEAR'}</span>
            </div>
            <div class="zone-box {get_zone_class('CENTER')}" style="flex: 1.4;">
                DRIVING LANE (CENTER)<br><span style="font-size: 0.75rem;">{'🚨 IN PATH' if 'CENTER' in zones_present else 'CLEAR'}</span>
            </div>
            <div class="zone-box {get_zone_class('RIGHT')}" style="flex: 1;">
                RIGHT ROADSIDE<br><span style="font-size: 0.75rem;">{'⚠️ ACTIVE' if 'RIGHT' in zones_present else 'CLEAR'}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics cards
        lighting_badge = "🌙 Night (MSRCR Active)" if is_night else "☀️ Daylight Mode"
        dist_proxy = f"{max_area * 100:.1f}% frame" if dets else "N/A"
        target_display = f"{active_animal} ({max_conf*100:.0f}%)" if active_animal else "None"
        
        metrics_placeholder.markdown(f"""
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 12px;">
            <div class="kpi-card">
                <div class="kpi-title">Detected Target</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #60a5fa;">{target_display}</div>
                <div class="kpi-subtitle">Zone: {active_zone or 'None'}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Proximity / Area Proxy</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #f59e0b;">{dist_proxy}</div>
                <div class="kpi-subtitle">BBox Fraction</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Environment Lighting</div>
                <div style="font-size: 1.1rem; font-weight: 700; color: #a78bfa;">{lighting_badge}</div>
                <div class="kpi-subtitle">Mean Brightness: {brightness:.1f}/255</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Active Targets in Frame</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #34d399;">{len(dets)} Animals</div>
                <div class="kpi-subtitle">Frame: #{frame_idx}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Frame detections table
        if dets:
            df_frame = pd.DataFrame(dets)[["class_name", "confidence", "zone", "area_fraction", "risk"]]
            df_frame["confidence"] = df_frame["confidence"].map(lambda x: f"{x*100:.1f}%")
            df_frame["area_fraction"] = df_frame["area_fraction"].map(lambda x: f"{x*100:.2f}%")
            alert_log_placeholder.dataframe(df_frame, use_container_width=True, hide_index=True)
        else:
            alert_log_placeholder.caption("No animal detected in current frame.")

        return annotated_rgb, dets

    # Initial frame render
    current_img, frame_dets = render_hud_state(st.session_state.current_frame)
    if current_img is not None:
        video_placeholder.image(current_img, use_container_width=True)

    # Active Playback Loop when "Play" is toggled
    if st.session_state.is_playing:
        cap_play = cv2.VideoCapture(selected_video_path)
        cap_play.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.current_frame)
        
        target_fps = fps * playback_speed
        frame_delay = max(0.01, 1.0 / target_fps)

        while st.session_state.is_playing and st.session_state.current_frame < total_frames - 1:
            step = frame_stride
            st.session_state.current_frame = min(total_frames - 1, st.session_state.current_frame + step)
            
            img_step, _ = render_hud_state(st.session_state.current_frame)
            if img_step is not None:
                video_placeholder.image(img_step, use_container_width=True)
            
            time.sleep(frame_delay)
            if st.session_state.current_frame >= total_frames - 1:
                st.session_state.is_playing = False
                st.rerun()


# =============================================================
# TAB 2: BI ANALYTICS PANEL
# =============================================================
elif nav_choice == "📊 BI Analytics Panel":
    st.title("📊 Driver Safety & Animal Detection BI Analytics Panel")
    st.markdown("Comprehensive operational intelligence: detections per session, risk severity distributions, and Day vs. Night comparative performance.")

    sessions = load_all_sessions()
    events_df = get_all_events_df(sessions)
    summary_df = get_sessions_summary_df(sessions)

    # 1. BI Executive Filter Bar
    st.markdown("##### 🔍 Global Filter Toolbar")
    f_col1, f_col2, f_col3, f_col4 = st.columns([3, 2, 2, 2])
    
    with f_col1:
        session_options = ["All Driving Sessions"] + list(summary_df["session_name"].unique())
        selected_session = st.selectbox("Select Driving Session", session_options)

    with f_col2:
        lighting_options = ["All Lighting Conditions", "Day", "Night"]
        selected_lighting = st.selectbox("Lighting Filter", lighting_options)

    with f_col3:
        risk_options = ["All Risk Levels", "High", "Medium", "Low"]
        selected_risk = st.selectbox("Risk Severity Filter", risk_options)

    with f_col4:
        all_animals = sorted(list(events_df["animal"].unique()))
        selected_animals = st.multiselect("Animal Class Filter", all_animals, default=[])

    # Apply filters
    filtered_events = events_df.copy()
    if selected_session != "All Driving Sessions":
        filtered_events = filtered_events[filtered_events["session_name"] == selected_session]
    if selected_lighting != "All Lighting Conditions":
        filtered_events = filtered_events[filtered_events["lighting"] == selected_lighting]
    if selected_risk != "All Risk Levels":
        filtered_events = filtered_events[filtered_events["risk"] == selected_risk]
    if selected_animals:
        filtered_events = filtered_events[filtered_events["animal"].isin(selected_animals)]

    # 2. Executive KPI Cards Ribbon
    total_sessions_count = len(summary_df) if selected_session == "All Driving Sessions" else 1
    total_detections_count = summary_df["total_detections"].sum() if selected_session == "All Driving Sessions" else summary_df[summary_df["session_name"] == selected_session]["total_detections"].sum()
    total_events_count = len(filtered_events)
    high_risk_count = len(filtered_events[filtered_events["risk"] == "High"])
    med_risk_count = len(filtered_events[filtered_events["risk"] == "Medium"])
    night_events_count = len(filtered_events[filtered_events["lighting"] == "Night"])
    night_ratio = (night_events_count / total_events_count * 100) if total_events_count > 0 else 0.0
    avg_conf_val = (filtered_events["confidence"].mean() * 100) if not filtered_events.empty else 0.0

    st.markdown("---")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #3b82f6;">
            <div class="kpi-title">Monitored Sessions</div>
            <div class="kpi-value">{total_sessions_count}</div>
            <div class="kpi-subtitle">Highway & Wildlife routes</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #10b981;">
            <div class="kpi-title">Total Animals Logged</div>
            <div class="kpi-value">{total_events_count}</div>
            <div class="kpi-subtitle">{total_detections_count} frame detections</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #ef4444;">
            <div class="kpi-title">High-Risk Threats</div>
            <div class="kpi-value">{high_risk_count}</div>
            <div class="kpi-subtitle">{med_risk_count} Medium risk cautions</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #8b5cf6;">
            <div class="kpi-title">Night Incident Share</div>
            <div class="kpi-value">{night_ratio:.1f}%</div>
            <div class="kpi-subtitle">{night_events_count} night threats logged</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi5:
        st.markdown(f"""
        <div class="kpi-card" style="border-left-color: #f59e0b;">
            <div class="kpi-title">Avg Model Confidence</div>
            <div class="kpi-value">{avg_conf_val:.1f}%</div>
            <div class="kpi-subtitle">YOLOv8 Detection Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 3. CORE CHART 1: DETECTIONS PER SESSION & TIMELINE
    st.markdown("### 📈 1. Detections & Incidents Per Session")
    
    chart_col1, chart_col2 = st.columns([6, 6])
    with chart_col1:
        # Bar chart: Detections and Events per session
        fig_sessions = go.Figure()
        fig_sessions.add_trace(go.Bar(
            x=summary_df["session_name"],
            y=summary_df["total_events"],
            name="Distinct Animal Incidents",
            marker_color="#3b82f6",
            text=summary_df["total_events"],
            textposition="auto"
        ))
        fig_sessions.add_trace(go.Bar(
            x=summary_df["session_name"],
            y=summary_df["high_risk"],
            name="High-Risk Warnings",
            marker_color="#ef4444",
            text=summary_df["high_risk"],
            textposition="auto"
        ))
        fig_sessions.update_layout(
            title="Incidents & High-Risk Alerts by Driving Session",
            barmode="group",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d1d5db"),
            xaxis=dict(tickangle=-20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=50, b=40)
        )
        st.plotly_chart(fig_sessions, use_container_width=True)

    with chart_col2:
        # Timeline scatter / frequency chart
        fig_time = px.scatter(
            filtered_events,
            x="start_time",
            y="animal",
            color="risk",
            size="duration_sec",
            color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
            hover_data=["session_name", "position", "confidence", "alert"],
            title="Chronological Incident Timeline (Start Time vs. Animal Encounter)"
        )
        fig_time.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d1d5db"),
            xaxis_title="Drive Timestamp (Seconds)",
            yaxis_title="Detected Animal Class",
            margin=dict(l=20, r=20, t=50, b=40)
        )
        st.plotly_chart(fig_time, use_container_width=True)

    # 4. CORE CHART 2: RISK-LEVEL DISTRIBUTION
    st.markdown("### ⚠️ 2. Risk-Level Distribution & Road Threat Analysis")
    r_col1, r_col2, r_col3 = st.columns([4, 4, 4])

    with r_col1:
        # Donut Chart: Risk Distribution
        risk_counts = filtered_events["risk"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Count"]
        color_map_donut = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"}
        fig_donut = px.pie(
            risk_counts,
            names="Risk",
            values="Count",
            hole=0.55,
            color="Risk",
            color_discrete_map=color_map_donut,
            title="Overall Threat Severity Breakdown"
        )
        fig_donut.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d1d5db"),
            margin=dict(l=20, r=20, t=50, b=40)
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with r_col2:
        # Stacked Bar: Risk by Animal Species
        fig_species = px.histogram(
            filtered_events,
            x="animal",
            color="risk",
            color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
            title="Risk Breakdown by Animal Species",
            barmode="stack"
        )
        fig_species.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d1d5db"),
            xaxis_title="Species",
            yaxis_title="Incident Count",
            margin=dict(l=20, r=20, t=50, b=40)
        )
        st.plotly_chart(fig_species, use_container_width=True)

    with r_col3:
        # Bar: Risk by Corridor Zone (Left vs Center vs Right)
        fig_zone = px.histogram(
            filtered_events,
            x="position",
            color="risk",
            color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
            title="Risk by Road Corridor Zone",
            barmode="group",
            category_orders={"position": ["LEFT", "CENTER", "RIGHT"]}
        )
        fig_zone.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d1d5db"),
            xaxis_title="Road Zone",
            yaxis_title="Incidents",
            margin=dict(l=20, r=20, t=50, b=40)
        )
        st.plotly_chart(fig_zone, use_container_width=True)

    # 5. CORE CHART 3: DAY VS. NIGHT COMPARISON
    st.markdown("### 🌙 vs. ☀️ 3. Day vs. Night Comparative Analysis")
    
    dn_kpi1, dn_kpi2, dn_kpi3, dn_kpi4 = st.columns(4)
    day_df = events_df[events_df["lighting"] == "Day"]
    night_df = events_df[events_df["lighting"] == "Night"]

    with dn_kpi1:
        st.metric(
            label="Day Incidents",
            value=f"{len(day_df)} Events",
            delta=f"{len(day_df[day_df['risk']=='High'])} High Risk",
            delta_color="inverse"
        )
    with dn_kpi2:
        st.metric(
            label="Night Incidents",
            value=f"{len(night_df)} Events",
            delta=f"{len(night_df[night_df['risk']=='High'])} High Risk",
            delta_color="inverse"
        )
    with dn_kpi3:
        day_conf = day_df["confidence"].mean() if not day_df.empty else 0.0
        night_conf = night_df["confidence"].mean() if not night_df.empty else 0.0
        st.metric(
            label="Day vs. Night Avg Confidence",
            value=f"{night_conf*100:.1f}% Night",
            delta=f"{(night_conf - day_conf)*100:.1f}% vs Day ({day_conf*100:.1f}%)"
        )
    with dn_kpi4:
        night_high_share = (len(night_df[night_df["risk"] == "High"]) / len(night_df) * 100) if len(night_df) > 0 else 0
        day_high_share = (len(day_df[day_df["risk"] == "High"]) / len(day_df) * 100) if len(day_df) > 0 else 0
        st.metric(
            label="Night Hazard Severity Rate",
            value=f"{night_high_share:.1f}%",
            delta=f"+{night_high_share - day_high_share:.1f}% vs Day ({day_high_share:.1f}%)",
            delta_color="inverse"
        )

    dn_col1, dn_col2 = st.columns([6, 6])
    with dn_col1:
        # Grouped bar chart: Species breakdown in Day vs Night
        fig_dn_species = px.histogram(
            events_df,
            x="animal",
            color="lighting",
            barmode="group",
            color_discrete_map={"Day": "#f59e0b", "Night": "#8b5cf6"},
            title="Animal Species Detected: Daylight vs. Nighttime"
        )
        fig_dn_species.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d1d5db"),
            xaxis_title="Species",
            yaxis_title="Incident Count",
            margin=dict(l=20, r=20, t=50, b=40)
        )
        st.plotly_chart(fig_dn_species, use_container_width=True)

    with dn_col2:
        # Risk distribution comparison between Day and Night
        fig_dn_risk = px.histogram(
            events_df,
            x="lighting",
            color="risk",
            barmode="stack",
            color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
            title="Risk Severity Proportions (Day vs. Night)"
        )
        fig_dn_risk.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#d1d5db"),
            xaxis_title="Lighting Condition",
            yaxis_title="Incident Count",
            margin=dict(l=20, r=20, t=50, b=40)
        )
        st.plotly_chart(fig_dn_risk, use_container_width=True)

    # 6. Detailed Session Event Audit Table & CSV Export
    st.markdown("---")
    st.markdown("### 📋 Detailed Incident Log & Data Export")
    
    export_df = filtered_events[["session_name", "lighting", "animal", "risk", "position", "start_time", "duration_sec", "confidence", "alert"]]
    st.dataframe(export_df, use_container_width=True, hide_index=True)

    dl_col1, dl_col2, _ = st.columns([2, 2, 4])
    with dl_col1:
        csv_data = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv_data,
            file_name="animal_detection_incidents.csv",
            mime="text/csv",
            use_container_width=True
        )
    with dl_col2:
        json_data = json.dumps(json.loads(export_df.to_json(orient="records")), indent=2)
        st.download_button(
            label="📥 Download Filtered Data (JSON)",
            data=json_data,
            file_name="animal_detection_incidents.json",
            mime="application/json",
            use_container_width=True
        )


# =============================================================
# TAB 3: SESSION MANAGER & LOGS
# =============================================================
elif nav_choice == "⚙️ Session Manager & Logs":
    st.title("⚙️ Session Manager & Video Batch Pipeline")
    st.markdown("Inspect registered sessions, rerun detection pipelines on new videos, and synchronize the analytics database.")

    sessions = load_all_sessions()
    summary_df = get_sessions_summary_df(sessions)

    st.markdown("#### 📁 Active Driving Sessions in Database")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### 🚀 Run Pipeline on New Video & Import to BI Analytics")
    
    all_videos = glob.glob("data/*.mp4")
    video_to_process = st.selectbox("Select Video Clip to Process", all_videos)
    session_title = st.text_input("Session Display Name", value=f"New Drive: {os.path.basename(video_to_process)}")
    session_lighting = st.selectbox("Lighting Environment", ["Night", "Day", "Twilight"])

    if st.button("⚡ Run Full Detection & Risk Pipeline", type="primary"):
        with st.status("Processing video through ADAS Pipeline...", expanded=True) as status:
            st.write("Initializing AnimalDetector and loading video...")
            from main import run_on_video
            out_vid = f"output/{os.path.basename(video_to_process)}"
            out_json = f"output/{os.path.splitext(os.path.basename(video_to_process))[0]}_detections.json"

            frames_log, events_log = run_on_video(video_to_process, out_vid, out_json)
            st.write(f"Processed {len(frames_log)} frames. Detected {len(events_log)} distinct animal events.")
            
            # Compute mean brightness
            cap_b = cv2.VideoCapture(video_to_process)
            ret_b, f_b = cap_b.read()
            cap_b.release()
            mb = float(np.mean(cv2.cvtColor(f_b, cv2.COLOR_BGR2GRAY))) if ret_b else 50.0

            # Import into analytics
            new_s = compute_session_metrics(
                session_id=f"session_{int(time.time())}",
                session_name=session_title,
                video_name=os.path.basename(video_to_process),
                lighting=session_lighting,
                mean_brightness=mb,
                fps=30.0,
                frame_logs=frames_log,
                events=events_log
            )
            add_or_update_session(new_s)
            status.update(label="✅ Processing complete & imported into BI Analytics!", state="complete")
            st.success(f"Successfully processed and registered session: **{session_title}**!")
            st.rerun()
