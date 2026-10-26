"""
main.py
--------
Primary Edge Runtime for:
- Module 2: Restricted Campus Zone Sentinel
- Module 3: Smart Hall & Auditorium Energy Wastage Sentinel
- Motion Gating Pipeline

Usage:
    python main.py --source 0                      # Live Laptop Webcam
    python main.py --source sample_video.mp4       # Recorded Footage
    python main.py --zones zones_multi.json        # Custom Multi-ROI

Press 'q' to quit the live feed.
"""
import argparse
import os
import cv2

import config
from detector import CampusEdgeSentinel
from logger import init_log, log_event


def parse_args():
    parser = argparse.ArgumentParser(description="Smart Edge CCTV Surveillance & Energy Control System")
    parser.add_argument("--source", default=None, help="Webcam index (0) or path to video file (terrace.mp4, classroom.mp4, etc.)")
    parser.add_argument("--preset", choices=["terrace", "classroom", "demo"], default=None,
                        help="Quick preset: 'terrace' (Restricted Zone), 'classroom' (Energy Saving), or 'demo'")
    parser.add_argument("--zones", default=None, help="Path to zone.json or zones_multi.json")
    parser.add_argument("--model", default=config.MODEL_PATH, help="YOLO model path")
    parser.add_argument("--disable-gating", action="store_true", help="Disable Motion Gating")
    parser.add_argument("--disable-mod2", action="store_true", help="Disable Module 2 (Restricted Zone)")
    parser.add_argument("--disable-mod3", action="store_true", help="Disable Module 3 (Energy Sentinel)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Preset / auto-detection logic
    source_str = str(args.source or "").lower()
    if args.preset == "terrace" or "terrace" in source_str:
        source = config.TERRACE_VIDEO
        zone_file = args.zones or config.ZONE_TERRACE_FILE
        enable_m2, enable_m3 = True, False
        print("[Smart Edge Preset] Loaded 'terrace.mp4' for Module 2: Restricted Zone Sentinel.")
    elif args.preset == "classroom" or "classroom" in source_str:
        source = config.CLASSROOM_VIDEO
        zone_file = args.zones or config.ZONE_FILE
        enable_m2, enable_m3 = False, True
        config.ROOM_NAME = "Smart Classroom 101"
        config.ZERO_OCCUPANCY_TIMEOUT_SECONDS = 1.5
        print("[Smart Edge Preset] Loaded 'classroom.mp4' for Module 3: Smart Classroom Energy Sentinel.")
    else:
        raw_source = args.source if args.source is not None else "0"
        source = int(raw_source) if raw_source.isdigit() else raw_source
        zone_file = args.zones or (config.ZONES_MULTI_FILE if os.path.exists(config.ZONES_MULTI_FILE) else config.ZONE_FILE)
        enable_m2 = not args.disable_mod2
        enable_m3 = not args.disable_mod3

    init_log()
    enable_gating = not args.disable_gating
    monitor = CampusEdgeSentinel(
        model_path=args.model,
        zone_path=zone_file,
        enable_gating=enable_gating
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Smart Edge] Error: Could not open video source '{args.source}'.")
        print("Tip: If running on a laptop without a camera, run test_demo_generator.py first.")
        return

    writer = None
    if config.ENABLE_VIDEO_RECORDING:
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(config.OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))
        print(f"[Smart Edge] Recording output to {config.OUTPUT_VIDEO_PATH}")

    print("=========================================================")
    print(" Smart Edge CCTV: Campus-Wide Surveillance & Energy Control")
    print(f" Source: {source} | Motion Gating: {enable_gating}")
    print(f" Module 2 (Restricted Zones): {enable_m2}")
    print(f" Module 3 (Energy Sentinel): {enable_m3}")
    print(" Press 'q' in the video window to quit.")
    print("=========================================================")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Smart Edge] End of video stream.")
            break

        annotated_frame, events = monitor.process_frame(
            frame,
            enable_module2=enable_m2,
            enable_module3=enable_m3
        )

        for event in events:
            if event.get("type") != "RELAY_CUTOFF":
                log_event(event, annotated_frame)
            else:
                print(f"[ENERGY ALERT] Automated Relay Cut-off! Saved ~{event.get('kwh_saved')} kWh.")

        if writer is not None:
            writer.write(annotated_frame)

        cv2.imshow("Smart Edge CCTV Monitor", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[Smart Edge] Saved recording: {config.OUTPUT_VIDEO_PATH}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
