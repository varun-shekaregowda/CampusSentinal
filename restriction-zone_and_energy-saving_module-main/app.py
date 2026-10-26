"""
app.py
-------
Unified Web Dashboard (Streamlit) for:
Smart Edge CCTV: Campus-Wide AI Surveillance & Automated Control System

Features:
- Live Edge CCTV Video Stream with AI HUD overlays
- Module 2: Restricted Zone Sentinel (Multi-ROI, directional entry, loitering)
- Module 3: Smart Hall & Auditorium Energy Wastage Sentinel (Occupancy, 0-occupancy timer, automated relay cut-off)
- Edge Motion Gating status & toggle
- Local Offline SQLite Event Log Viewer & Analytics Charts
"""
import os
import time
import cv2
import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np

import config
import detector
import importlib
importlib.reload(detector)
from detector import CampusEdgeSentinel
from database import (
    init_db,
    fetch_recent_security_events,
    fetch_recent_energy_events,
    fetch_energy_summary
)
from logger import init_log, log_event

# Configure Streamlit page
st.set_page_config(
    page_title="Smart Edge CCTV | Campus AI Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        text-align: center;
    }
    .alert-banner {
        background-color: #FEE2E2;
        border-left: 5px solid #EF4444;
        color: #991B1B;
        padding: 12px;
        border-radius: 4px;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .success-banner {
        background-color: #DCFCE7;
        border-left: 5px solid #22C55E;
        color: #166534;
        padding: 12px;
        border-radius: 4px;
        font-weight: 600;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
init_log()
init_db()

# --- Sidebar Controls ---
st.sidebar.image("https://img.icons8.com/fluency/96/cctv.png", width=64)
st.sidebar.title("Edge Node Settings")
st.sidebar.markdown("**Hardware:** Edge CPU (Quantized YOLOv8-Nano)")

source_choice = st.sidebar.selectbox(
    "Video Feed Source",
    [
        "Terrace Restricted Zone (terrace.mp4)",
        "Smart Classroom Energy Sentinel (classroom.mp4)",
        "Campus Synthetic Demo (sample_campus_demo.mp4)",
        "Laptop Webcam (0)",
        "Custom Video File"
    ]
)

if "Terrace" in source_choice:
    video_source = config.TERRACE_VIDEO
    default_mod2, default_mod3 = True, False
    if os.path.exists(config.ZONE_CUSTOM_FILE):
        active_zone_path = config.ZONE_CUSTOM_FILE
    elif os.path.exists(config.ZONE_TERRACE_FILE):
        active_zone_path = config.ZONE_TERRACE_FILE
    else:
        active_zone_path = config.ZONE_FILE
    default_timeout = 5.0
elif "Classroom" in source_choice:
    video_source = config.CLASSROOM_VIDEO
    default_mod2, default_mod3 = False, True
    active_zone_path = config.ZONE_FILE
    config.ROOM_NAME = "Smart Classroom 101"
    default_timeout = 1.5
elif "Campus" in source_choice:
    demo_file = os.path.join(config.BASE_DIR, "sample_campus_demo.mp4")
    if not os.path.exists(demo_file):
        import test_demo_generator
        test_demo_generator.generate_demo_video()
    video_source = demo_file
    default_mod2, default_mod3 = True, True
    active_zone_path = config.ZONES_MULTI_FILE if os.path.exists(config.ZONES_MULTI_FILE) else config.ZONE_FILE
    default_timeout = 3.0
elif "Webcam" in source_choice:
    video_source = 0
    default_mod2, default_mod3 = True, True
    active_zone_path = config.ZONES_MULTI_FILE if os.path.exists(config.ZONES_MULTI_FILE) else config.ZONE_FILE
    default_timeout = 5.0
    st.sidebar.caption("📷 *Note: If camera feed is black, slide open the physical privacy shutter on your laptop bezel.*")
else:
    video_source = st.sidebar.text_input("Enter video file path", "terrace.mp4")
    default_mod2, default_mod3 = True, True
    active_zone_path = config.ZONE_FILE
    default_timeout = 5.0

st.sidebar.markdown("---")
st.sidebar.subheader("Module Controls")
enable_mod2 = st.sidebar.checkbox("Module 2: Restricted Zone Sentinel", value=default_mod2)
enable_mod3 = st.sidebar.checkbox("Module 3: Energy Wastage Sentinel", value=default_mod3)
enable_gating = st.sidebar.checkbox("Motion Gating (Edge CPU Saver)", value=True)
is_monitoring_active = st.session_state.get("run_monitoring_toggle", False)

# --- Flexible Custom Restricted Zone Designer ---
with st.sidebar.expander("🖱️ Draw Restricted Zone (PrtScn Style)", expanded=True):
    st.markdown("Click the button below to open the drawing window. **Click and drag your mouse across any area, then press ENTER on your keyboard.**")

    col_btn1, col_btn2 = st.columns([1.5, 1])
    with col_btn1:
        if st.button("✂️ Draw Zone on Video", use_container_width=True):
            is_cam = str(video_source).isdigit() or (isinstance(video_source, int) and video_source >= 0)
            if is_monitoring_active and is_cam:
                st.warning("⚠️ Turn OFF '🟢 Start Edge Sentinel' first so the camera can be used for drawing.")
            else:
                import subprocess
                import sys
                cmd = [sys.executable, "zone_setup.py", "--source", str(video_source)]
                subprocess.Popen(cmd)
                st.info("👉 Window opened in front! Drag your mouse across the area, then press ENTER.")

    with col_btn2:
        if st.button("🔄 Refresh Zone", use_container_width=True):
            st.session_state["use_custom_zone"] = True
            st.rerun()

    # Zone selection: Default vs Custom Drawn
    has_custom = os.path.exists(config.ZONE_CUSTOM_FILE)
    zone_options = ["Custom Drawn Zone", "Default Preset Zone"] if has_custom else ["Default Preset Zone"]
    selected_zone_type = st.radio("Active Zone Source", zone_options, index=0)

    if selected_zone_type == "Custom Drawn Zone" and has_custom:
        active_zone_path = config.ZONE_CUSTOM_FILE
        st.success("✅ Using **Your Drawn Custom Zone**")
        # Visual thumbnail preview of user drawn zone
        try:
            source_arg = int(video_source) if str(video_source).isdigit() else video_source
            is_cam = isinstance(source_arg, int)
            ret_prev, sample_prev = False, None
            if not is_cam:
                t_cap = cv2.VideoCapture(video_source)
                ret_prev, sample_prev = t_cap.read()
                t_cap.release()
            elif not is_monitoring_active:
                t_cap = cv2.VideoCapture(source_arg, cv2.CAP_DSHOW)
                ret_prev, sample_prev = t_cap.read()
                t_cap.release()

            if ret_prev and sample_prev is not None:
                import json
                with open(config.ZONE_CUSTOM_FILE, "r", encoding="utf-8") as f:
                    z_json = json.load(f)
                c_pts = np.array(z_json["zones"][0]["points"], dtype=np.int32)
                prev = sample_prev.copy()
                cv2.polylines(prev, [c_pts], True, (0, 0, 255), 3)
                overlay = prev.copy()
                cv2.fillPoly(overlay, [c_pts], (0, 0, 255))
                cv2.addWeighted(overlay, 0.25, prev, 0.75, 0, prev)
                st.image(cv2.cvtColor(prev, cv2.COLOR_BGR2RGB), caption="Preview of Marked Zone (Red)", use_container_width=True)
        except Exception:
            pass
    else:
        st.info("Using default camera preset zone.")

with st.sidebar.expander("⚙️ Fine-Tuning Parameters"):
    timeout_sec = st.slider("0-Occupancy Timeout (sec)", 1.0, 30.0, float(default_timeout))
    config.ZERO_OCCUPANCY_TIMEOUT_SECONDS = timeout_sec
    illum_thresh = st.slider("Illumination Threshold", 30.0, 150.0, float(config.ACTIVE_ILLUMINATION_THRESHOLD))
    config.ACTIVE_ILLUMINATION_THRESHOLD = illum_thresh
    room_watts = st.number_input("Room Wattage Load (Watts)", value=float(config.ROOM_LIGHT_WATTS), step=100.0)
    config.ROOM_LIGHT_WATTS = room_watts

st.sidebar.markdown("---")
run_monitoring = st.sidebar.toggle("🟢 Start Edge Sentinel", value=False, key="run_monitoring_toggle")

# --- Header Section ---
st.markdown('<div class="main-title">🛡️ Smart Edge CCTV: Campus-Wide AI Surveillance</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">100% On-Device Edge Pipeline • Sub-50ms Latency • Automated Energy Control • Offline SQLite Logging</div>', unsafe_allow_html=True)

# Fetch latest summary metrics
summary = fetch_energy_summary()
total_saved_kwh = summary.get("total_kwh_saved", 0.0)
cost_saved = total_saved_kwh * config.COST_PER_KWH

# KPI Metric Row
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Edge Node Status", "ACTIVE" if run_monitoring else "STANDBY", delta="CPU Inference")
with kpi2:
    st.metric("Automated Cut-Offs", f"{summary.get('total_cutoffs', 0)} Events", delta="Power Guarded")
with kpi3:
    st.metric("Total Energy Saved", f"{total_saved_kwh:.3f} kWh", delta=f"${cost_saved:.2f} Saved")
with kpi4:
    st.metric("Bandwidth Reduction", "> 95%", delta="Edge JSON Metadata")

st.markdown("---")

# Main Interface: Video stream + Real-time Telemetry
col_video, col_telemetry = st.columns([1.8, 1.2])

with col_video:
    st.subheader("📹 Live Edge Camera Feed")
    video_placeholder = st.empty()

with col_telemetry:
    st.subheader("⚡ Real-time Telemetry & Relay")
    telemetry_placeholder = st.empty()

# Bottom Tabs: Historical SQLite Logs and Energy Charts
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📋 Security Intrusion Logs", "💡 Energy Wastage & Relay Logs", "📊 Energy Savings Analytics"])

with tab1:
    sec_logs_placeholder = st.empty()
    st.markdown("#### 📸 Captured Intruder Snapshot Evidence")
    if os.path.exists(config.SNAPSHOT_DIR):
        snap_files = [os.path.join(config.SNAPSHOT_DIR, f) for f in os.listdir(config.SNAPSHOT_DIR) if f.endswith(('.jpg', '.png'))]
        snap_files.sort(key=os.path.getmtime, reverse=True)
        if snap_files:
            snap_cols = st.columns(min(4, len(snap_files)))
            for i, s_file in enumerate(snap_files[:4]):
                with snap_cols[i]:
                    st.image(s_file, caption=os.path.basename(s_file), use_container_width=True)
        else:
            st.caption("No snapshots captured yet. When an intruder enters, full-scene and cropped portraits will appear here.")
    else:
        st.caption("Snapshots folder initialized.")

with tab2:
    energy_logs_placeholder = st.empty()

with tab3:
    chart_placeholder = st.empty()

# --- Edge Execution Loop ---
if run_monitoring:
    zone_path = active_zone_path
    monitor = CampusEdgeSentinel(
        zone_path=zone_path,
        enable_gating=enable_gating
    )

    source_val = int(video_source) if str(video_source).isdigit() else video_source
    if isinstance(source_val, int):
        cap = cv2.VideoCapture(source_val, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source_val)
    else:
        cap = cv2.VideoCapture(source_val)

    if not cap.isOpened():
        st.error(f"Could not open video feed from '{video_source}'.")
    else:
        frame_count = 0
        while run_monitoring:
            ret, frame = cap.read()
            if not ret:
                # If reading video file, loop continuously for live demo
                if isinstance(video_source, str) and os.path.exists(video_source):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                else:
                    st.warning("Video stream ended.")
                    break

            annotated_frame, events = monitor.process_frame(
                frame,
                enable_module2=enable_mod2,
                enable_module3=enable_mod3
            )

            # Log events
            for event in events:
                if event.get("type") != "RELAY_CUTOFF":
                    log_event(event, annotated_frame)

            # Render video frame to Streamlit
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Render Telemetry Card
            with telemetry_placeholder.container():
                lights_on = getattr(monitor, "lights_on", False)
                fans_rotating = getattr(monitor, "fans_rotating", False)
                illum = getattr(monitor, "current_illumination", 0.0)
                relay_cut_off = getattr(monitor, "relay_cut_off", False)
                occ_count = getattr(monitor, "occupancy_count", 0)
                energy_status = getattr(monitor, "energy_status", "NORMAL")

                if enable_mod3:
                    if occ_count == 0:
                        actions_li = []
                        if lights_on:
                            actions_li.append(f"<li style='color:#f87171; font-weight:bold;'>💡 <b>Turn OFF Lights</b> (Lights are ON - Luminance: {illum:.1f})</li>")
                        else:
                            actions_li.append("<li style='color:#4ade80;'>💡 Lights are OFF</li>")

                        if fans_rotating:
                            actions_li.append("<li style='color:#f87171; font-weight:bold;'>🌀 <b>Turn OFF Ceiling Fans</b> (Ceiling Fans are actively rotating)</li>")
                        else:
                            actions_li.append("<li style='color:#4ade80;'>🌀 Ceiling Fans are stopped</li>")

                        actions_html = "".join(actions_li)

                        if relay_cut_off:
                            st.markdown(f"""
                            <div style="background-color: rgba(239, 68, 68, 0.2); border: 2px solid #ef4444; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                                <div style="font-size: 1.12rem; font-weight: bold; color: #f87171;">⚠️ NO ONE IN CLASSROOM! (0 Occupants)</div>
                                <div style="margin-top: 6px; font-size: 0.92rem; color: #fecaca;">
                                    <b>Identified Wastage & Directives:</b>
                                    <ul style="margin: 4px 0 6px 18px;">
                                        {actions_html}
                                    </ul>
                                </div>
                                <div style="background: rgba(0,0,0,0.35); padding: 6px 10px; border-radius: 4px; font-size: 0.84rem; color: #38bdf8;">
                                    ⚡ <b>Automated Cut-Off Triggered:</b> Relays isolated power to prevent wattage leakage.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            rem = max(0.0, config.ZERO_OCCUPANCY_TIMEOUT_SECONDS - (time.time() - (monitor.empty_start_time or time.time())))
                            st.markdown(f"""
                            <div style="background-color: rgba(245, 158, 11, 0.2); border: 2px solid #f59e0b; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
                                <div style="font-size: 1.12rem; font-weight: bold; color: #fbbf24;">⏳ NO ONE IN CLASSROOM! (0 Occupants)</div>
                                <div style="margin-top: 6px; font-size: 0.92rem; color: #fde68a;">
                                    <b>Action Required:</b>
                                    <ul style="margin: 4px 0 6px 18px;">
                                        {actions_html}
                                    </ul>
                                </div>
                                <div style="background: rgba(0,0,0,0.35); padding: 6px 10px; border-radius: 4px; font-size: 0.84rem; color: #fbbf24;">
                                    ⏱️ Grace countdown: <b>{rem:.1f}s</b> remaining before automated relay cut-off.
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="success-banner">
                            ✅ <b>Status: Classroom Active & Occupied</b><br>
                            Room Occupancy: <b>{occ_count} Person(s)</b> | Lights: {"ON" if lights_on else "OFF"} | Fans: {"ROTATING" if fans_rotating else "STOPPED"}
                        </div>
                        """, unsafe_allow_html=True)

                    c_m1, c_m2 = st.columns(2)
                    with c_m1:
                        st.metric("Live Occupancy", f"{occ_count} Person(s)")
                        st.metric("Ceiling Fans", "ROTATING" if fans_rotating else "STOPPED",
                                  delta="Wasting Power" if (occ_count == 0 and fans_rotating) else None,
                                  delta_color="inverse")
                    with c_m2:
                        st.metric("Classroom Lights", f"ON ({illum:.0f})" if lights_on else "OFF",
                                  delta="Wasting Power" if (occ_count == 0 and lights_on) else None,
                                  delta_color="inverse")
                        st.metric("Power Cut-Off Relay", "CUT-OFF (OPEN)" if relay_cut_off else "NORMAL (CLOSED)")
                else:
                    # Security Module 2 Telemetry
                    has_intrusion = any(e.get("type") in ["ZONE_ENTRY", "DIRECTIONAL_ENTRY", "LOITERING"] for e in events)
                    if has_intrusion:
                        st.markdown("""
                        <div class="alert-banner">
                            🚨 <b>PERIMETER INTRUSION DETECTED!</b><br>
                            Unauthorized target detected within restricted zone. Snapshot evidence captured.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="success-banner">
                            🛡️ <b>Restricted Zone Perimeter: SECURE</b><br>
                            Sentinel AI actively scanning human trajectory & boundaries.
                        </div>
                        """, unsafe_allow_html=True)

                    c_m1, c_m2 = st.columns(2)
                    with c_m1:
                        st.metric("Active Zones", f"{len(monitor.zones)} Zone(s)")
                        st.metric("Tracked Persons", f"{len(monitor.track_states)} Target(s)")
                    with c_m2:
                        snap_count = len(os.listdir(config.SNAPSHOT_DIR)) if os.path.exists(config.SNAPSHOT_DIR) else 0
                        st.metric("Snapshots Captured", f"{snap_count} Photos")
                        st.metric("Edge Processing", "YOLOv8 CPU + Gating")

            # Update database tables every 10 frames
            frame_count += 1
            if frame_count % 10 == 0:
                sec_records = fetch_recent_security_events(limit=15)
                if sec_records:
                    sec_df = pd.DataFrame(sec_records)
                    sec_logs_placeholder.dataframe(
                        sec_df[["timestamp", "zone_name", "event_type", "track_id", "detail", "severity"]],
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    sec_logs_placeholder.info("No security intrusion events logged yet.")

                energy_records = fetch_recent_energy_events(limit=15)
                if energy_records:
                    en_df = pd.DataFrame(energy_records)
                    energy_logs_placeholder.dataframe(
                        en_df[["timestamp", "room_name", "status", "occupancy_count", "empty_duration_sec", "kwh_saved", "detail"]],
                        use_container_width=True,
                        hide_index=True
                    )

                    # Analytics chart
                    fig = px.bar(
                        en_df.head(10),
                        x="timestamp",
                        y="kwh_saved",
                        color="occupancy_count",
                        title="Recent Energy Cut-Off Power Savings (kWh)",
                        labels={"kwh_saved": "Power Saved (kWh)", "timestamp": "Timestamp"},
                        color_continuous_scale="Viridis"
                    )
                    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
                    chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"live_chart_{frame_count}")
                else:
                    energy_logs_placeholder.info("No energy wastage events logged yet.")

            time.sleep(0.01)

        cap.release()
else:
    # Standby view - show existing database records and guidance
    st.info("👆 Click **'🟢 Start Edge Sentinel'** in the sidebar to start live monitoring and AI control.")

    sec_records = fetch_recent_security_events(limit=15)
    if sec_records:
        sec_df = pd.DataFrame(sec_records)
        sec_logs_placeholder.dataframe(sec_df, use_container_width=True, hide_index=True)
    else:
        sec_logs_placeholder.info("SQLite Database ready at 'campus_edge.db'. No security events recorded yet.")

    energy_records = fetch_recent_energy_events(limit=15)
    if energy_records:
        en_df = pd.DataFrame(energy_records)
        energy_logs_placeholder.dataframe(en_df, use_container_width=True, hide_index=True)

        fig = px.bar(
            en_df,
            x="timestamp",
            y="kwh_saved",
            title="Recorded Power Savings from Automated Relay Cut-offs (kWh)",
            labels={"kwh_saved": "Power Saved (kWh)", "timestamp": "Timestamp"}
        )
        chart_placeholder.plotly_chart(fig, use_container_width=True, key="standby_analytics_chart")
    else:
        energy_logs_placeholder.info("SQLite Database ready at 'campus_edge.db'. No energy cut-off events recorded yet.")
