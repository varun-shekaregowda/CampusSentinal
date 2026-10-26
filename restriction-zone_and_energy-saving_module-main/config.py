"""
Central configuration for Smart Edge CCTV:
- Module 2: Restricted Campus Zone Sentinel (Perimeter monitoring, directional entry, false-alarm rejection)
- Module 3: Smart Hall & Auditorium Energy Wastage Sentinel (Occupancy counter, illumination check, 0-occupancy relay trigger)
- Motion Gating (Edge CPU optimization via MOG2 background subtractor)
- Local Offline Storage: SQLite database
"""
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Edge AI & Model Settings ---
MODEL_PATH = "yolov8n.pt"      # Pretrained YOLOv8-Nano (CPU optimized)
CONFIDENCE_THRESHOLD = 0.30    # Detection confidence threshold

# --- Motion Gating (Edge CPU power saver) ---
ENABLE_MOTION_GATING = True    # Skip YOLO on static frames to save edge CPU
MOTION_MIN_AREA = 800          # Minimum contour area for motion detection
MOTION_HEARTBEAT_FRAMES = 25   # Force YOLO check every N frames even without motion

# --- Module 2: Restricted Campus Zone Sentinel Settings ---
ZONE_FILE = os.path.join(BASE_DIR, "zone.json")
ZONES_MULTI_FILE = os.path.join(BASE_DIR, "zones_multi.json")
ZONE_TERRACE_FILE = os.path.join(BASE_DIR, "zone_terrace.json")
ZONE_CUSTOM_FILE = os.path.join(BASE_DIR, "zone_custom.json")

# Sample Video Presets
TERRACE_VIDEO = os.path.join(BASE_DIR, "terrace.mp4")
CLASSROOM_VIDEO = os.path.join(BASE_DIR, "classroom.mp4")

# False alarm rejection parameters (filters wind-blown debris, leaves, small shadows)
MIN_PERSON_BBOX_AREA = 1200       # Minimum bounding box area in pixels
MAX_PERSON_BBOX_AREA = 250000     # Maximum bounding box area (reject massive shadow artifacts)
MIN_HUMAN_ASPECT_RATIO = 1.1      # Height / Width ratio (humans are generally taller than wide)
MAX_HUMAN_ASPECT_RATIO = 4.5      # Maximum height-to-width ratio

# Alert & loitering timing
ALERT_COOLDOWN_SECONDS = 4.0      # Cooldown between repeat alerts for same track
LOITER_THRESHOLD_SECONDS = 3.0    # Seconds inside zone to trigger loitering alert
MIN_DIRECTIONAL_DISPLACEMENT = 15 # Minimum pixel displacement to establish entry vector

# --- Module 3: Smart Hall & Auditorium Energy Wastage Sentinel Settings ---
ROOM_NAME = "Main Auditorium / Lecture Hall"
ROOM_LIGHT_WATTS = 2500.0         # 2.5 kW total load
LIGHT_WATTS = 1200.0              # 1.2 kW classroom tube lights
FAN_WATTS = 300.0                 # 0.3 kW ceiling fans
COST_PER_KWH = 0.15               # Cost per kWh in dollars/currency
ACTIVE_ILLUMINATION_THRESHOLD = 65.0 # Average luminance threshold (0-255) for "Lights ON"
ZERO_OCCUPANCY_TIMEOUT_SECONDS = 5.0 # Seconds of 0-occupancy before automated relay cut-off (demo friendly)
ENABLE_RELAY_CUTOFF = True        # Auto cut-off relay trigger

# --- Local Offline Storage (SQLite Database) ---
DB_PATH = os.path.join(BASE_DIR, "campus_edge.db")
LOG_FILE = os.path.join(BASE_DIR, "intrusion_log.csv")
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")

# --- Video Output Recording ---
ENABLE_VIDEO_RECORDING = False
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "output_recording.mp4")

# --- Telegram Alerts (Optional) ---
ENABLE_TELEGRAM = False
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""
