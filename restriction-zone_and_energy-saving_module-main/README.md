# Restricted Zone Intrusion Detection System

A computer vision prototype that detects when a person enters a marked
restricted zone, using YOLOv8 for real-time person detection and tracking,
and OpenCV for the zone-intersection logic. Runs on sample/recorded footage
or a laptop webcam — **no CCTV installation required.**

## Features

- **Real-time person detection & tracking** — YOLOv8 (pretrained, no training required), with persistent IDs across frames
- **Custom zone definition** — click-based tool to mark any polygonal restricted area on your own camera feed
- **Entry alerts** — visual overlay + on-screen banner the moment someone steps into the zone
- **Loitering detection** — escalates an alert if a person remains in the zone beyond a configurable time threshold
- **Event logging** — every entry/loitering event is timestamped, logged to CSV, and paired with a saved snapshot image
- **Works on recorded footage or live webcam** — same code, just swap the `--source` argument
- **Live web dashboard** (Streamlit) — video feed, live event counters, and a recent-events table in the browser
- **Telegram alerts** — optional real-time push notification (with photo) the moment an intrusion occurs
- **Output video recording** — saves the fully annotated run to an `.mp4`, as a demo backup if a live run has issues

## Architecture

```
Video / Webcam Input
        |
        v
Person Detection + Tracking (YOLOv8)
        |
        v
Zone Intersection Check (point-in-polygon, per tracked person)
        |
        +--> Alert Overlay (entry / loitering banner)
        |
        +--> Event Log (CSV + snapshot image)
```

## Project structure

```
.
├── config.py           # all tunable settings in one place
├── zone_setup.py       # interactive tool to mark the restricted zone
├── detector.py         # core detection, tracking, and zone logic
├── logger.py           # CSV logging + snapshot saving + Telegram trigger
├── telegram_alert.py   # optional Telegram push notifications
├── main.py             # entry point / run loop (OpenCV window + video recording)
├── app.py              # Streamlit web dashboard
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

The first run auto-downloads the YOLOv8 nano weights (~6 MB) — no manual
model download needed. An internet connection is only required the first time.

## Usage

### Step 1 — Define your restricted zone
Run this once per camera/video setup:

```bash
python zone_setup.py --source 0                # webcam
python zone_setup.py --source sample_video.mp4  # video file
```

Click points around the restricted area (minimum 3), press `s` to save.
This creates `zone.json`.

### Step 2 — Run the detector

```bash
python main.py --source 0                # webcam
python main.py --source sample_video.mp4  # recorded footage
```

- **Green box** — person detected, outside the zone
- **Red box + "INTRUSION DETECTED"** — person inside the zone
- Press `q` to quit

Every entry and loitering event is written to `intrusion_log.csv`, with a
matching snapshot saved in `snapshots/`.

## Running the Streamlit dashboard

```bash
streamlit run app.py
```

This opens a browser tab with a live video feed, entry/loitering counters,
and a table of recent events — reusing the same detection engine as
`main.py`. Make sure `zone.json` already exists (run `zone_setup.py` first).

## Enabling Telegram alerts (optional)

1. In Telegram, message **@BotFather** → `/newbot` → follow the prompts →
   copy the bot token.
2. Message **@userinfobot** → `/start` → copy your numeric chat ID.
3. Send your new bot at least one message first (bots can't message you
   until you've messaged them once).
4. In `config.py`, set:
   ```python
   ENABLE_TELEGRAM = True
   TELEGRAM_BOT_TOKEN = "your-token-here"
   TELEGRAM_CHAT_ID = "your-chat-id-here"
   ```

Every entry/loitering event will now also arrive as a Telegram message
with a snapshot photo attached.

## Output video recording

Enabled by default (`ENABLE_VIDEO_RECORDING = True` in `config.py`). Each
run of `main.py` saves the fully annotated video to `output_recording.mp4`
— useful as a polished artifact to submit, or as a fallback if your live
demo has technical issues.

## Configuration

All tunable parameters live in `config.py`:

| Setting | Purpose | Default |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | Minimum detection confidence | `0.4` |
| `ALERT_COOLDOWN_SECONDS` | Minimum gap between repeat alerts | `5` |
| `LOITER_THRESHOLD_SECONDS` | Time in zone before a loitering alert | `3` |
| `MODEL_PATH` | YOLO weights to use | `yolov8n.pt` |

## Tech stack

- **Python 3.8+**
- **YOLOv8** (Ultralytics) — pretrained person detection + tracking
- **OpenCV** — video I/O, drawing, point-in-polygon geometry
- **CSV** — lightweight event logging (no database required for a prototype)

## Demo tips (for a live hackathon presentation)

1. Test the zone setup on your presentation laptop *before* going on stage — lighting changes can shift where you want to draw the zone.
2. Have a short recorded video as a fallback in case the live webcam demo has technical issues.
3. Walk into the zone slowly and stay a few seconds to show off both the entry alert and the loitering escalation.

## Future scope

- Direction-aware detection (flag only actual entries, not edge loitering)
- Multi-zone and multi-camera support
- Email/SMS alert integration
- Face recognition layer to identify *who* entered, not just that someone did
