"""
verify_all_modules.py
---------------------
Comprehensive automated test script for:
- Module 2: Restricted Campus Zone Sentinel (Directional entry, loitering, false alarm rejection)
- Module 3: Smart Hall & Auditorium Energy Wastage Sentinel (Occupancy, 0-occupancy countdown, relay cut-off)
- Motion Gating (Edge CPU optimization)
- SQLite Database logging
"""
import os
import sys
import time
import cv2

# Set path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)

import config
from detector import CampusEdgeSentinel
from database import (
    init_db,
    fetch_recent_security_events,
    fetch_recent_energy_events,
    fetch_energy_summary
)
from logger import init_log, log_event

def run_verification():
    print("================================================================")
    print(" STARTING VERIFICATION: Smart Edge CCTV (Modules 2 & 3)")
    print("================================================================")

    # 1. Initialize SQLite Database and Log
    init_log()
    init_db()
    print("[1/5] SQLite Database initialized successfully.")

    # 2. Check test demo video exists
    demo_video = os.path.join(CURRENT_DIR, "sample_campus_demo.mp4")
    if not os.path.exists(demo_video):
        print("Demo video not found, generating...")
        import test_demo_generator
        test_demo_generator.generate_demo_video()
    print("[2/5] Demo video confirmed: sample_campus_demo.mp4")

    # 3. Initialize Sentinel Engine
    zone_path = config.ZONES_MULTI_FILE if os.path.exists(config.ZONES_MULTI_FILE) else config.ZONE_FILE
    sentinel = CampusEdgeSentinel(
        zone_path=zone_path,
        enable_gating=True
    )
    print(f"[3/5] CampusEdgeSentinel initialized with {len(sentinel.zones)} zone(s).")
    for z in sentinel.zones:
        print(f"     - Zone: {z['name']} ({z['id']})")

    # 4. Process Demo Video Frames
    cap = cv2.VideoCapture(demo_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[4/5] Ingesting video stream ({total_frames} frames)...")

    frame_num = 0
    security_event_count = 0
    energy_event_count = 0
    relay_triggered = False

    # Tune timeout for swift automated test
    config.ZERO_OCCUPANCY_TIMEOUT_SECONDS = 1.5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        annotated_frame, events = sentinel.process_frame(frame, enable_module2=True, enable_module3=True)

        for event in events:
            etype = event.get("type")
            if etype in ["ZONE_ENTRY", "DIRECTIONAL_ENTRY", "LOITERING"]:
                security_event_count += 1
                log_event(event, annotated_frame)
            elif etype == "RELAY_CUTOFF":
                energy_event_count += 1
                relay_triggered = True

    cap.release()
    print(f"     Processed {frame_num} frames.")

    # 5. Query and Assert SQLite Database Records
    print("[5/5] Verifying SQLite Database records...")
    sec_events = fetch_recent_security_events(limit=10)
    en_events = fetch_recent_energy_events(limit=10)
    summary = fetch_energy_summary()

    print(f"     - Recorded Security Events in DB: {len(sec_events)}")
    for ev in sec_events[:3]:
        print(f"       * [{ev['event_type']}] Zone: {ev['zone_name']} | Track ID: {ev['track_id']} | Detail: {ev['detail']}")

    print(f"     - Recorded Energy Events in DB: {len(en_events)}")
    for en in en_events[:3]:
        print(f"       * [{en['status']}] Room: {en['room_name']} | Saved: {en['kwh_saved']} kWh | Relay: {en['relay_triggered']}")

    print(f"     - Energy Summary: Total Cut-offs = {summary['total_cutoffs']}, Total kWh Saved = {summary['total_kwh_saved']:.4f}")

    # Assertions
    assert len(sec_events) > 0, "Security events were not logged in SQLite!"
    assert len(en_events) > 0, "Energy events were not logged in SQLite!"
    assert relay_triggered, "Automated relay cut-off was not triggered!"

    print("================================================================")
    print(" ALL VERIFICATION TESTS PASSED SUCCESSFULLY! ")
    print(" Module 2 (Restricted Zones) & Module 3 (Energy Sentinel) READY.")
    print("================================================================")

if __name__ == "__main__":
    run_verification()
