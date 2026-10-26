"""
zone_setup.py
--------------
Interactive Drag-and-Select Tool for Restricted Zones (PrtScn / Snipping Tool Style).
Supports click-and-drag mouse rectangle selection with real-time visual feedback.

Controls:
    - Click and DRAG mouse across any area on the video/camera frame
    - Release mouse when area is covered
    - Press ENTER or SPACE to save
    - Press 'r' to re-drag / redraw
    - Press 'c' or ESC to cancel
"""
import argparse
import json
import os
import cv2
import config

# Global state for mouse drag
drag_start = None
drag_end = None
is_dragging = False
selected_rect = None


def mouse_drag_callback(event, x, y, flags, param):
    global drag_start, drag_end, is_dragging, selected_rect

    if event == cv2.EVENT_LBUTTONDOWN:
        drag_start = (x, y)
        drag_end = (x, y)
        is_dragging = True
        selected_rect = None

    elif event == cv2.EVENT_MOUSEMOVE and is_dragging:
        drag_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP and is_dragging:
        drag_end = (x, y)
        is_dragging = False
        if drag_start is not None:
            x1, y1 = drag_start
            x2, y2 = drag_end
            xmin, xmax = min(x1, x2), max(x1, x2)
            ymin, ymax = min(y1, y2), max(y1, y2)
            if (xmax - xmin) > 15 and (ymax - ymin) > 15:
                selected_rect = (xmin, ymin, xmax - xmin, ymax - ymin)


def select_zone_drag(source=0, output_file=None):
    global drag_start, drag_end, is_dragging, selected_rect
    drag_start = None
    drag_end = None
    is_dragging = False
    selected_rect = None

    if output_file is None:
        output_file = config.ZONE_CUSTOM_FILE

    is_webcam = str(source).isdigit() or (isinstance(source, int) and source >= 0)
    source_val = int(source) if str(source).isdigit() else source

    if is_webcam:
        # On Windows, DirectShow (CAP_DSHOW) prevents MSMF -1072875772 capture errors
        cap = cv2.VideoCapture(source_val, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source_val)
    else:
        cap = cv2.VideoCapture(source_val)

    if not cap.isOpened():
        print(f"[Zone Setup] Error: Could not open source '{source}'. If live monitoring is running, please stop it first.")
        return None

    # Camera sensor warmup (wait for valid frame)
    frame = None
    for _ in range(25):
        ret, temp_frame = cap.read()
        if ret and temp_frame is not None:
            frame = temp_frame
            break

    if frame is None:
        print(f"[Zone Setup] Error: Could not read frame from source '{source}'.")
        cap.release()
        return None

    win_name = "DRAG TO SELECT RESTRICTED ZONE (PrtScn Style)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    # Force window to top of all other windows (so it's not hidden behind Chrome)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow(win_name, min(1280, frame.shape[1]), min(720, frame.shape[0]))
    cv2.setMouseCallback(win_name, mouse_drag_callback)

    saved_points = None

    while True:
        # If webcam and not currently dragging or selected, update live frame
        if is_webcam and selected_rect is None and not is_dragging:
            ret, fresh_frame = cap.read()
            if ret and fresh_frame is not None:
                frame = fresh_frame

        display = frame.copy()

        # Check if camera frame is pitch black (physical privacy slider closed)
        if is_webcam and float(frame.mean()) < 3.0 and selected_rect is None and not is_dragging:
            h_disp, w_disp = display.shape[:2]
            card_w, card_h = min(580, w_disp - 30), 135
            cx1 = max(10, (w_disp - card_w) // 2)
            cy1 = max(50, (h_disp - card_h) // 2)
            cx2 = cx1 + card_w
            cy2 = cy1 + card_h

            cv2.rectangle(display, (cx1, cy1), (cx2, cy2), (25, 25, 25), -1)
            cv2.rectangle(display, (cx1, cy1), (cx2, cy2), (0, 165, 255), 2)

            cv2.putText(display, "! CAMERA FEED IS BLACK (0 LUX)", (cx1 + 15, cy1 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 215, 255), 2)
            cv2.putText(display, "1. Open physical privacy slider switch on laptop screen bezel", (cx1 + 15, cy1 + 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            cv2.putText(display, "2. Check keyboard privacy key (e.g. F10 / Fn+F10 / F4)", (cx1 + 15, cy1 + 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            cv2.putText(display, "• Drag mouse to draw zone anyway, or press 's' for demo scene", (cx1 + 15, cy1 + 108),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1)

        # Instructions banner at top
        cv2.rectangle(display, (0, 0), (display.shape[1], 40), (20, 20, 20), -1)
        mode_hint = "LIVE WEBCAM" if is_webcam else "VIDEO"
        cv2.putText(display, f"[{mode_hint}] CLICK & DRAG with Mouse | ENTER: Save | 's': Demo Scene | ESC: Cancel",
                    (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)

        # Draw live dragging box
        if is_dragging and drag_start and drag_end:
            cv2.rectangle(display, drag_start, drag_end, (0, 255, 0), 2)
            overlay = display.copy()
            cv2.rectangle(overlay, drag_start, drag_end, (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.25, display, 0.75, 0, display)

        # Draw confirmed selected box
        elif selected_rect is not None:
            sx, sy, sw, sh = selected_rect
            cv2.rectangle(display, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 3)
            overlay = display.copy()
            cv2.rectangle(overlay, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.30, display, 0.70, 0, display)
            cv2.putText(display, f"SELECTED: {sw}x{sh}px - Press ENTER to Save", (sx + 10, max(sy - 10, 60)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow(win_name, display)
        key = cv2.waitKey(20) & 0xFF

        # ENTER or SPACE to save
        if key in [13, 32]:
            if selected_rect is not None:
                sx, sy, sw, sh = selected_rect
                custom_points = [
                    [int(sx), int(sy)],
                    [int(sx + sw), int(sy)],
                    [int(sx + sw), int(sy + sh)],
                    [int(sx), int(sy + sh)]
                ]
                zone_data = {
                    "zones": [{
                        "id": "Zone-Custom",
                        "name": "User Dragged Restricted Zone",
                        "points": custom_points,
                        "color": [0, 0, 255]
                    }]
                }
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(zone_data, f, indent=2)

                # Update zone_terrace.json and zone.json so all modes use the drawn zone
                if "terrace" in str(source).lower() or os.path.exists(config.ZONE_TERRACE_FILE):
                    with open(config.ZONE_TERRACE_FILE, "w", encoding="utf-8") as f:
                        json.dump(zone_data, f, indent=2)

                with open(config.ZONE_FILE, "w", encoding="utf-8") as f:
                    json.dump({"zone": custom_points}, f, indent=2)

                print(f"[Zone Setup] Success! Saved custom zone to '{output_file}' and '{config.ZONE_TERRACE_FILE}'.")
                saved_points = custom_points
                break
            else:
                print("[Zone Setup] Please click and drag an area first before pressing ENTER.")

        # Switch to demo scene background
        elif key == ord('s'):
            demo_bg = config.TERRACE_VIDEO if os.path.exists(config.TERRACE_VIDEO) else config.CLASSROOM_VIDEO
            t_cap = cv2.VideoCapture(demo_bg)
            t_ret, t_fr = t_cap.read()
            t_cap.release()
            if t_ret and t_fr is not None:
                frame = cv2.resize(t_fr, (display.shape[1], display.shape[0]))
                is_webcam = False
                print("[Zone Setup] Switched to sample background scene.")

        # Reset selection
        elif key == ord('r'):
            selected_rect = None
            drag_start = None
            drag_end = None
            is_dragging = False
            print("[Zone Setup] Selection cleared. Drag a new area.")

        # Cancel / Quit
        elif key in [27, ord('c'), ord('q')]:
            print("[Zone Setup] Cancelled.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return saved_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Webcam index (0) or path to video file (terrace.mp4, etc.)")
    parser.add_argument("--output", default=config.ZONE_CUSTOM_FILE, help="Path to save zone JSON")
    args = parser.parse_args()

    select_zone_drag(source=args.source, output_file=args.output)


if __name__ == "__main__":
    main()
