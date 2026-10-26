"""
logger.py
----------
Unified event dispatcher and logger:
- Writes to local SQLite database (campus_edge.db)
- Saves incident snapshots to snapshots/ directory
- Maintains CSV log backup
- Sends optional real-time Telegram push notifications
"""
import csv
import os
from datetime import datetime
import cv2

import config
from database import init_db, log_security_event_db
from telegram_alert import send_telegram_alert


def init_log():
    os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)
    init_db()
    if not os.path.exists(config.LOG_FILE):
        with open(config.LOG_FILE, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp", "event_type", "track_id", "zone", "detail", "snapshot"])


def log_event(event: dict, frame) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    event_type = event.get("type", "ALERT")
    track_id = event.get("track_id", 0)
    zone_name = event.get("zone_name", "Perimeter")
    detail = event.get("detail", event.get("duration", ""))

    ts = int(datetime.now().timestamp())
    snapshot_name = f"full_{event_type.lower()}_id{track_id}_{ts}.jpg"
    snapshot_path = os.path.join(config.SNAPSHOT_DIR, snapshot_name)
    cv2.imwrite(snapshot_path, frame)

    # Dedicated cropped portrait snapshot of the intruder person
    crop_path = ""
    bbox = event.get("bbox", None)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        fh, fw = frame.shape[:2]
        pad = 20
        cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
        cx2, cy2 = min(fw, x2 + pad), min(fh, y2 + pad)
        crop_img = frame[cy1:cy2, cx1:cx2]
        if crop_img.size > 0:
            crop_name = f"intruder_id{track_id}_{ts}.jpg"
            crop_path = os.path.join(config.SNAPSHOT_DIR, crop_name)
            cv2.imwrite(crop_path, crop_img)

    # 1. Log to SQLite (stores snapshot path for evidence)
    log_security_event_db(
        event_type=event_type,
        track_id=track_id,
        zone_id=event.get("zone_id", "Zone-1"),
        zone_name=zone_name,
        detail=str(detail),
        snapshot_path=crop_path if crop_path else snapshot_path,
        severity="HIGH" if event_type in ["ZONE_ENTRY", "DIRECTIONAL_ENTRY"] else "MEDIUM"
    )

    # 2. Log to CSV backup
    with open(config.LOG_FILE, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([timestamp, event_type, track_id, zone_name, detail, snapshot_path])

    print(f"[{event_type}] ID {track_id} in {zone_name} at {timestamp} ({detail})")

    # 3. Optional Telegram Notification
    alert_message = (
        f"\U0001F6A8 Smart Edge Alert: {event_type}\n"
        f"Zone: {zone_name}\n"
        f"Track ID: {track_id}\n"
        f"Detail: {detail}\n"
        f"Time: {timestamp}"
    )
    send_telegram_alert(alert_message, snapshot_path)
