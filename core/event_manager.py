import sqlite3
import os
import datetime
import cv2

class EventManager:
    def __init__(self, db_path="storage/db/campus.sqlite3"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                cam_id TEXT,
                event_type TEXT,
                description TEXT,
                screenshot_path TEXT,
                nvr_playback_url TEXT
            )
        ''')
        # Migration: Add video_path if it doesn't exist
        try:
            c.execute('ALTER TABLE events ADD COLUMN video_path TEXT')
        except sqlite3.OperationalError:
            pass # Column already exists
            
        conn.commit()
        conn.close()

    def log_event(self, cam_id, event_type, description, frame=None, video_path=""):
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        screenshot_path = ""
        
        import re
        if frame is not None:
            filename = f"incident_{cam_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            screenshot_path = re.sub(r'[\\/]+', '/', f"storage/screenshots/{filename}")
            os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
            cv2.imwrite(screenshot_path, frame)
            
        video_path = re.sub(r'[\\/]+', '/', video_path) if video_path else ""
        
        # For NVR fetch, log the exact UTC timestamp.
        nvr_playback_url = f"nvr_fetch://{cam_id}?time={timestamp.replace(' ', 'T')}Z"
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            INSERT INTO events (timestamp, cam_id, event_type, description, screenshot_path, nvr_playback_url, video_path)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, cam_id, event_type, description, screenshot_path, nvr_playback_url, video_path))
        conn.commit()
        event_id = c.lastrowid
        conn.close()
        
        return {
            "id": event_id,
            "timestamp": timestamp,
            "cam_id": cam_id,
            "event_type": event_type,
            "description": description,
            "screenshot": screenshot_path,
            "video": video_path
        }

event_manager = EventManager()
