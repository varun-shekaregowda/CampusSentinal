"""
database.py
-----------
Local offline SQLite database manager for Smart Edge CCTV.
Stores security perimeter intrusion logs and energy wastage/relay logs
locally without requiring internet connectivity.
"""
import sqlite3
import os
from datetime import datetime
import config

def get_connection():
    os.makedirs(os.path.dirname(config.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(config.DB_PATH, timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # Security alerts table (Module 2)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS security_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        zone_id TEXT DEFAULT 'Zone-1',
        zone_name TEXT DEFAULT 'Restricted Zone',
        event_type TEXT NOT NULL,
        track_id INTEGER,
        detail TEXT,
        snapshot_path TEXT,
        severity TEXT DEFAULT 'HIGH',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Energy wastage and automated relay cut-off events (Module 3)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS energy_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
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
    """)

    # Cumulative energy analytics table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS energy_summary (
        id INTEGER PRIMARY KEY CHECK (id = 1),
        total_cutoffs INTEGER DEFAULT 0,
        total_seconds_saved REAL DEFAULT 0.0,
        total_kwh_saved REAL DEFAULT 0.0,
        last_updated TEXT
    )
    """)

    cursor.execute("""
    INSERT OR IGNORE INTO energy_summary (id, total_cutoffs, total_seconds_saved, total_kwh_saved, last_updated)
    VALUES (1, 0, 0.0, 0.0, datetime('now'))
    """)

    conn.commit()
    conn.close()

def log_security_event_db(event_type: str, track_id: int, zone_id="Zone-1", zone_name="Restricted Zone", detail="", snapshot_path="", severity="HIGH"):
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO security_events (timestamp, zone_id, zone_name, event_type, track_id, detail, snapshot_path, severity)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (now_str, zone_id, zone_name, event_type, track_id, str(detail), snapshot_path, severity))
    conn.commit()
    conn.close()

def log_energy_event_db(room_name: str, status: str, occupancy: int, illumination: float, empty_duration: float, kwh_saved: float, relay_triggered: bool, detail=""):
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
    INSERT INTO energy_events (timestamp, room_name, status, occupancy_count, illumination_val, empty_duration_sec, kwh_saved, relay_triggered, detail)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (now_str, room_name, status, occupancy, illumination, empty_duration, kwh_saved, 1 if relay_triggered else 0, detail))

    if relay_triggered and kwh_saved > 0:
        cursor.execute("""
        UPDATE energy_summary
        SET total_cutoffs = total_cutoffs + 1,
            total_seconds_saved = total_seconds_saved + ?,
            total_kwh_saved = total_kwh_saved + ?,
            last_updated = ?
        WHERE id = 1
        """, (empty_duration, kwh_saved, now_str))

    conn.commit()
    conn.close()

def fetch_recent_security_events(limit=30):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT timestamp, zone_name, event_type, track_id, detail, snapshot_path, severity
    FROM security_events ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def fetch_recent_energy_events(limit=30):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT timestamp, room_name, status, occupancy_count, illumination_val, empty_duration_sec, kwh_saved, relay_triggered, detail
    FROM energy_events ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def fetch_energy_summary():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT total_cutoffs, total_seconds_saved, total_kwh_saved, last_updated FROM energy_summary WHERE id = 1")
    row = cursor.fetchone()
    conn.close()
    if row:
        return dict(row)
    return {"total_cutoffs": 0, "total_seconds_saved": 0.0, "total_kwh_saved": 0.0, "last_updated": ""}
