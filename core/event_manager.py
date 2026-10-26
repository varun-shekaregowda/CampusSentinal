import sqlite3
import os
import datetime
import cv2

class EventManager:
    def __init__(self, db_path="storage/db/campus.sqlite3"):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = self._get_connection()
        c = conn.cursor()

        # 1. Main Unified Events Table (Malpractice, Alerts, Overarching Log)
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
        try:
            c.execute('ALTER TABLE events ADD COLUMN video_path TEXT')
        except sqlite3.OperationalError:
            pass # Column already exists

        # 2. Security Perimeter Tracking Events (Module 2: Restricted Zones)
        c.execute('''
            CREATE TABLE IF NOT EXISTS security_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cam_id TEXT DEFAULT '',
                zone_id TEXT DEFAULT 'Zone-1',
                zone_name TEXT DEFAULT 'Restricted Zone',
                event_type TEXT NOT NULL,
                track_id INTEGER,
                detail TEXT,
                snapshot_path TEXT,
                severity TEXT DEFAULT 'HIGH',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 3. Energy Wastage & Relay Events (Module 3: Smart Classrooms)
        c.execute('''
            CREATE TABLE IF NOT EXISTS energy_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cam_id TEXT DEFAULT '',
                room_name TEXT NOT NULL,
                status TEXT NOT NULL,
                occupancy_count INTEGER NOT NULL,
                illumination_val REAL NOT NULL,
                empty_duration_sec REAL DEFAULT 0.0,
                kwh_saved REAL DEFAULT 0.0,
                relay_triggered INTEGER DEFAULT 0,
                detail TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # 4. Cumulative Energy Analytics Summary
        c.execute('''
            CREATE TABLE IF NOT EXISTS energy_summary (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                total_cutoffs INTEGER DEFAULT 0,
                total_seconds_saved REAL DEFAULT 0.0,
                total_kwh_saved REAL DEFAULT 0.0,
                last_updated TEXT
            )
        ''')
        c.execute('''
            INSERT OR IGNORE INTO energy_summary (id, total_cutoffs, total_seconds_saved, total_kwh_saved, last_updated)
            VALUES (1, 0, 0.0, 0.0, datetime('now'))
        ''')

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
        nvr_playback_url = f"nvr_fetch://{cam_id}?time={timestamp.replace(' ', 'T')}Z"
        
        conn = self._get_connection()
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

    def log_security_event(self, cam_id: str, event_type: str, track_id: int, zone_id: str = "Zone-1",
                           zone_name: str = "Restricted Zone", detail: str = "", frame=None,
                           bbox=None, severity: str = "HIGH", video_path: str = ""):
        """Logs perimeter intrusion / loitering events into security_events and main events tables."""
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        snapshot_path = ""
        import re

        if frame is not None:
            ts_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"sec_{cam_id}_id{track_id}_{ts_str}.jpg"
            snapshot_path = re.sub(r'[\\/]+', '/', f"storage/screenshots/{filename}")
            os.makedirs(os.path.dirname(snapshot_path), exist_ok=True)
            cv2.imwrite(snapshot_path, frame)

            crop_path = ""
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                fh, fw = frame.shape[:2]
                pad = 20
                cx1, cy1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
                cx2, cy2 = min(fw, int(x2) + pad), min(fh, int(y2) + pad)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size > 0:
                    crop_name = f"sec_crop_{cam_id}_id{track_id}_{ts_str}.jpg"
                    crop_path = re.sub(r'[\\/]+', '/', f"storage/screenshots/{crop_name}")
                    cv2.imwrite(crop_path, crop)

        conn = self._get_connection()
        c = conn.cursor()
        c.execute('''
            INSERT INTO security_events (timestamp, cam_id, zone_id, zone_name, event_type, track_id, detail, snapshot_path, severity)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, cam_id, zone_id, zone_name, event_type, track_id, detail, snapshot_path, severity))
        conn.commit()
        conn.close()

        # Also route to main events table for unified WebSocket broadcast & incident clip recording
        main_alert = self.log_event(
            cam_id=cam_id,
            event_type=event_type,
            description=f"[{zone_name}] ID {track_id}: {detail}",
            frame=None,  # Frame already saved above
            video_path=video_path
        )
        main_alert["screenshot"] = snapshot_path
        main_alert["screenshot_path"] = snapshot_path
        if crop_path:
            main_alert["portrait_path"] = crop_path

        if snapshot_path:
            conn = self._get_connection()
            c = conn.cursor()
            c.execute("UPDATE events SET screenshot_path = ? WHERE id = ?", (snapshot_path, main_alert["id"]))
            conn.commit()
            conn.close()

        return main_alert

    def log_energy_event(self, cam_id: str, room_name: str, status: str, occupancy: int,
                         illumination: float, empty_duration: float, kwh_saved: float,
                         relay_triggered: bool, detail: str = "", frame=None):
        """Logs energy wastage and automated relay cut-off events."""
        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_connection()
        c = conn.cursor()

        c.execute('''
            INSERT INTO energy_events (timestamp, cam_id, room_name, status, occupancy_count, illumination_val, empty_duration_sec, kwh_saved, relay_triggered, detail)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (timestamp, cam_id, room_name, status, occupancy, illumination, empty_duration, kwh_saved, 1 if relay_triggered else 0, detail))

        if relay_triggered and kwh_saved > 0:
            c.execute('''
                UPDATE energy_summary
                SET total_cutoffs = total_cutoffs + 1,
                    total_seconds_saved = total_seconds_saved + ?,
                    total_kwh_saved = total_kwh_saved + ?,
                    last_updated = ?
                WHERE id = 1
            ''', (empty_duration, kwh_saved, timestamp))

        conn.commit()
        conn.close()

        # Route major energy wastage cut-offs to main events table for live UI alerting
        return self.log_event(
            cam_id=cam_id,
            event_type="ENERGY_RELAY_CUTOFF" if relay_triggered else "ENERGY_WASTAGE",
            description=f"[{room_name}] {detail} (Saved ~{kwh_saved:.4f} kWh)",
            frame=frame
        )

    def fetch_recent_security_events(self, limit: int = 30):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('''
            SELECT id, timestamp, cam_id, zone_id, zone_name, event_type, track_id, detail, snapshot_path, severity
            FROM security_events ORDER BY id DESC LIMIT ?
        ''', (limit,))
        rows = c.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def fetch_recent_energy_events(self, limit: int = 30):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('''
            SELECT id, timestamp, cam_id, room_name, status, occupancy_count, illumination_val, empty_duration_sec, kwh_saved, relay_triggered, detail
            FROM energy_events ORDER BY id DESC LIMIT ?
        ''', (limit,))
        rows = c.fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def fetch_energy_summary(self):
        conn = self._get_connection()
        c = conn.cursor()
        c.execute('SELECT total_cutoffs, total_seconds_saved, total_kwh_saved, last_updated FROM energy_summary WHERE id = 1')
        row = c.fetchone()
        conn.close()
        if row:
            return dict(row)
        return {"total_cutoffs": 0, "total_seconds_saved": 0.0, "total_kwh_saved": 0.0, "last_updated": ""}

event_manager = EventManager()
