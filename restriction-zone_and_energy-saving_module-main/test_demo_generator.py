"""
test_demo_generator.py
----------------------
Generates a realistic synthetic test video (sample_campus_demo.mp4)
demonstrating both:
  1. Module 2: Directional restricted zone intrusion + leaf/debris false alarm rejection
  2. Module 3: Active room lighting -> occupancy drops to 0 -> countdown -> automated relay cut-off!

Zero external video files needed; completely self-contained for testing and project demos.
"""
import os
import cv2
import numpy as np

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "sample_campus_demo.mp4")
WIDTH, HEIGHT = 640, 480
FPS = 20
DURATION_SEC = 16
TOTAL_FRAMES = FPS * DURATION_SEC


def draw_human_figure(img, cx, cy, scale=1.0, shirt_color=(40, 160, 220), pants_color=(50, 50, 50)):
    """Draws a recognizable standing human silhouette for YOLOv8 person detection."""
    # Head
    head_r = int(14 * scale)
    head_cy = cy - int(90 * scale)
    cv2.circle(img, (cx, head_cy), head_r, (200, 190, 180), -1)

    # Torso (Shirt)
    torso_w = int(24 * scale)
    torso_h = int(50 * scale)
    torso_x1 = cx - torso_w // 2
    torso_y1 = head_cy + head_r
    torso_x2 = cx + torso_w // 2
    torso_y2 = torso_y1 + torso_h
    cv2.rectangle(img, (torso_x1, torso_y1), (torso_x2, torso_y2), shirt_color, -1)

    # Arms
    cv2.line(img, (torso_x1, torso_y1 + 5), (torso_x1 - int(10 * scale), torso_y2 - 10), shirt_color, max(2, int(6 * scale)))
    cv2.line(img, (torso_x2, torso_y1 + 5), (torso_x2 + int(10 * scale), torso_y2 - 10), shirt_color, max(2, int(6 * scale)))

    # Legs (Pants)
    leg_w = max(2, int(8 * scale))
    cv2.line(img, (cx - 6, torso_y2), (cx - 8, cy), pants_color, leg_w)
    cv2.line(img, (cx + 6, torso_y2), (cx + 8, cy), pants_color, leg_w)


def generate_demo_video():
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, FPS, (WIDTH, HEIGHT))

    print(f"[Generator] Generating {DURATION_SEC}s test demo video to: {os.path.basename(OUTPUT_FILE)}")

    for frame_idx in range(TOTAL_FRAMES):
        t = frame_idx / float(FPS)
        frame = np.full((HEIGHT, WIDTH, 3), 215, dtype=np.uint8)  # Bright room floor/wall

        # Draw auditorium stage / background room structure
        cv2.rectangle(frame, (0, 0), (WIDTH, 140), (180, 190, 200), -1)  # Wall
        cv2.line(frame, (0, 140), (WIDTH, 140), (120, 130, 140), 3)       # Baseboard

        # Restricted Yard marking on floor (for visual context)
        pts = np.array([[160, 100], [440, 100], [440, 420], [160, 420]], dtype=np.int32)
        cv2.polylines(frame, [pts], True, (150, 150, 220), 2)
        cv2.putText(frame, "Restricted Zone (Floor Grid)", (170, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 160), 1)

        # Simulation Phases:
        # Phase 1: t = 0s to 5s -> Person 1 walks into restricted zone, loiters
        # Phase 2: t = 5s to 8s -> Person 2 present in hall; small leaves blowing (false alarm check)
        # Phase 3: t = 8s to 16s -> Hall empty, lights ON -> 0-occupancy countdown -> relay cut-off!

        # 1. Person 1 (Restricted Zone Intruder)
        if t <= 5.5:
            # Person walks from outside (x=80) into restricted zone (x=260)
            px = int(min(270, 80 + t * 45))
            py = int(320 + np.sin(t * 6) * 5)
            draw_human_figure(frame, px, py, scale=1.4, shirt_color=(50, 120, 210))

        # 2. Person 2 (Auditorium Hall Attendee)
        if 2.0 <= t <= 7.5:
            # Sits/stands in right side of room, then exits to the right
            p2_t = t - 2.0
            if p2_t < 4.0:
                p2_x = 520
            else:
                p2_x = int(520 + (p2_t - 4.0) * 100)  # walks off screen
            draw_human_figure(frame, p2_x, 340, scale=1.3, shirt_color=(40, 170, 80))

        # 3. Small wind-blown debris / leaves (Tests False Alarm Filter)
        if 3.0 <= t <= 7.0:
            leaf_x = int(100 + (t - 3.0) * 80)
            leaf_y = int(220 + np.cos(t * 8) * 15)
            # Small green/brown irregular spot
            cv2.circle(frame, (leaf_x, leaf_y), 5, (30, 120, 40), -1)
            cv2.ellipse(frame, (leaf_x + 4, leaf_y + 2), (6, 3), 30, 0, 360, (20, 90, 30), -1)

        # 4. Auditorium Lighting Simulation for Module 3
        # If t > 13.0, simulate relay cut-off darkening the room
        if t > 13.5:
            # Lights turned off by automated relay cut-off!
            frame = (frame * 0.20).astype(np.uint8)
            cv2.putText(frame, "[SIMULATED RELAY: LIGHTS POWER CUT-OFF]", (120, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 120, 255), 2)

        out.write(frame)

    out.release()
    print(f"[Generator] Success! Created '{os.path.basename(OUTPUT_FILE)}' ({TOTAL_FRAMES} frames).")
    return OUTPUT_FILE


if __name__ == "__main__":
    generate_demo_video()
