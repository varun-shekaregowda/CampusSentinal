import multiprocessing as mp
import uvicorn
from fastapi import FastAPI, WebSocket, Request, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List, Tuple, Any
import cv2
import asyncio
import json
import time
import os
import shutil

from core.config_manager import config_manager
from core.inference import run_camera_process

os.makedirs("storage", exist_ok=True)
os.makedirs("storage/screenshots", exist_ok=True)
os.makedirs("storage/clips", exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="ui/static"), name="static")
app.mount("/storage", StaticFiles(directory="storage"), name="storage")
app.mount("/screenshots", StaticFiles(directory="storage/screenshots"), name="screenshots")
app.mount("/clips", StaticFiles(directory="storage/clips"), name="clips")
templates = Jinja2Templates(directory="ui/templates")

frame_queues = {}
command_queues = {}
status_queues = {}
cached_camera_status = {}
active_processes = {}
alert_queue = mp.Queue()
active_connections = []

class CameraCreate(BaseModel):
    name: str
    url: str
    mode: str
    roi: Optional[List[Any]] = None

class CameraUpdate(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    mode: Optional[str] = None
    roi: Optional[List[Any]] = None

def start_camera_process(cam):
    cam_id = cam['id']
    if cam_id in active_processes:
        stop_camera_process(cam_id)
        
    fq = mp.Queue(maxsize=1) # Keep size small for lowest latency
    cq = mp.Queue()
    sq = mp.Queue(maxsize=1)
    frame_queues[cam_id] = fq
    command_queues[cam_id] = cq
    status_queues[cam_id] = sq
    p = mp.Process(target=run_camera_process, args=(cam, fq, alert_queue, cq, sq), daemon=True)
    p.start()
    active_processes[cam_id] = p
    print(f"Started Inference Process for {cam_id}")

def stop_camera_process(cam_id):
    if cam_id in active_processes:
        p = active_processes[cam_id]
        if cam_id in command_queues:
            try:
                command_queues[cam_id].put({"action": "shutdown"})
            except:
                pass
        p.join(timeout=0.6)
        if p.is_alive():
            try:
                p.terminate()
            except:
                pass
            p.join(timeout=0.5)
            if p.is_alive():
                try:
                    p.kill()
                    p.join(timeout=0.3)
                except:
                    pass
        del active_processes[cam_id]
        print(f"Stopped Inference Process for {cam_id}")
    if cam_id in frame_queues:
        del frame_queues[cam_id]
    if cam_id in command_queues:
        del command_queues[cam_id]
    if cam_id in status_queues:
        del status_queues[cam_id]
    if cam_id in cached_camera_status:
        del cached_camera_status[cam_id]

@app.on_event("startup")
async def startup_event():
    for cam in config_manager.get_cameras():
        start_camera_process(cam)
    asyncio.create_task(broadcast_alerts())

async def broadcast_alerts():
    """
    WebSocket Telemetry & Alert Broadcaster:
    - Pushes real-time alerts enriched with occupancy, lux, fan status, and active alerts
    - Periodically (every 1.0s) broadcasts live camera telemetry to connected clients
    """
    last_telemetry_time = 0.0
    while True:
        # 1. Process Incident Alerts from camera inference workers
        if not alert_queue.empty():
            alert = alert_queue.get_nowait()
            cam_id = alert.get("cam_id")
            
            # Enrich alert payload with the latest camera telemetry
            cached_stat = cached_camera_status.get(cam_id, {})
            alert["occupancy_count"] = cached_stat.get("occupancy", cached_stat.get("occupancy_count", 0))
            alert["lux_level"] = cached_stat.get("lux_level", 0.0)
            alert["fans_active"] = cached_stat.get("fans_active", False)
            alert["lights_on"] = cached_stat.get("lights_on", False)
            alert["relay_cut_off"] = cached_stat.get("relay_cut_off", False)
            alert["kwh_saved"] = cached_stat.get("kwh_saved", 0.0)
            alert["energy_status"] = cached_stat.get("energy_status", "NORMAL")
            alert["active_alerts"] = cached_stat.get("active_alerts", [])

            message = json.dumps(alert)
            for connection in list(active_connections):
                try:
                    await connection.send_text(message)
                except:
                    if connection in active_connections:
                        active_connections.remove(connection)

        # 2. Periodic Live Telemetry Broadcast (Every 1.0s)
        now = time.time()
        if now - last_telemetry_time >= 1.0:
            last_telemetry_time = now
            for cam_id, sq in list(status_queues.items()):
                try:
                    stat = None
                    while not sq.empty():
                        stat = sq.get_nowait()
                    if stat:
                        cached_camera_status[cam_id] = stat

                    cached = cached_camera_status.get(cam_id)
                    if cached and active_connections:
                        telemetry_msg = json.dumps({
                            "type": "telemetry",
                            "cam_id": cam_id,
                            "occupancy_count": cached.get("occupancy", 0),
                            "countdown": cached.get("countdown", 0.0),
                            "intrusion_count": cached.get("intrusion_count", 0),
                            "any_intrusion": cached.get("any_intrusion", False),
                            "lux_level": cached.get("lux_level", 0.0),
                            "fans_active": cached.get("fans_active", False),
                            "lights_on": cached.get("lights_on", False),
                            "relay_cut_off": cached.get("relay_cut_off", False),
                            "kwh_saved": cached.get("kwh_saved", 0.0),
                            "energy_status": cached.get("energy_status", "NORMAL"),
                            "active_alerts": cached.get("active_alerts", []),
                            "is_capturing": cached.get("is_capturing", False),
                            "capture_left": cached.get("capture_left", 0),
                            "fps": cached.get("fps", 20)
                        })
                        for connection in list(active_connections):
                            try:
                                await connection.send_text(telemetry_msg)
                            except:
                                if connection in active_connections:
                                    active_connections.remove(connection)
                except:
                    pass

        await asyncio.sleep(0.05)

@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# --- CRUD APIs ---
@app.get("/api/cameras")
async def api_get_cameras():
    return config_manager.get_cameras()

@app.get("/api/events")
async def api_get_events():
    import sqlite3
    db_path = "storage/db/campus.sqlite3"
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM events ORDER BY id DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    events = []
    for ix in rows:
        d = dict(ix)
        d["screenshot"] = d.get("screenshot_path", "")
        d["video"] = d.get("video_path", "")
        events.append(d)
    return events

@app.post("/api/upload_video")
async def api_upload_video(file: UploadFile = File(...)):
    os.makedirs("storage", exist_ok=True)
    file_path = os.path.join("storage", file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"url": file_path}

@app.post("/api/cameras")
async def api_add_camera(cam: CameraCreate):
    new_cam = config_manager.add_camera(cam.name, cam.url, cam.mode, cam.roi)
    start_camera_process(new_cam)
    return new_cam

@app.put("/api/cameras/{cam_id}")
async def api_update_camera(cam_id: str, updates: CameraUpdate):
    updated_cam = config_manager.update_camera(cam_id, updates.dict(exclude_unset=True))
    if not updated_cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    # Terminate the existing camera process and spawn a new CameraPipeline
    stop_camera_process(cam_id)
    await asyncio.sleep(0.3)
    start_camera_process(updated_cam)
    return updated_cam

@app.post("/api/config")
@app.put("/api/config")
async def api_save_config(request: Request):
    data = await request.json()
    if isinstance(data, dict):
        cam_id = data.get("id") or data.get("cam_id")
        if cam_id:
            updated = config_manager.update_camera(cam_id, data)
            if updated:
                stop_camera_process(cam_id)
                await asyncio.sleep(0.3)
                start_camera_process(updated)
                return {"status": "success", "camera": updated}
    elif isinstance(data, list):
        for cam in data:
            cam_id = cam.get("id")
            if cam_id:
                updated = config_manager.update_camera(cam_id, cam)
                if updated:
                    stop_camera_process(cam_id)
                    await asyncio.sleep(0.2)
                    start_camera_process(updated)
        return {"status": "success"}
    return {"status": "ignored"}

@app.delete("/api/cameras/{cam_id}")
async def api_delete_camera(cam_id: str):
    config_manager.delete_camera(cam_id)
    stop_camera_process(cam_id)
    return {"status": "deleted"}

@app.post("/api/clear_warning/{cam_id}/{track_id}")
async def api_clear_warning(cam_id: str, track_id: int):
    if cam_id in command_queues:
        command_queues[cam_id].put({"action": "clear", "track_id": track_id})
        return {"status": "cleared"}
    raise HTTPException(status_code=404, detail="Camera not found")

@app.post("/api/cameras/{cam_id}/control")
async def api_camera_control(cam_id: str, request: Request):
    if cam_id not in command_queues:
        raise HTTPException(status_code=404, detail="Camera not found")
    data = await request.json()
    command_queues[cam_id].put(data)
    return {"status": "success"}

@app.get("/api/cameras/{cam_id}/status")
async def api_camera_status(cam_id: str):
    default_status = {
        "frame": 0,
        "total": 0,
        "paused": False,
        "speed": 1.0,
        "is_capturing": False,
        "capture_left": 0,
        "occupancy": 0,
        "occupancy_count": 0,
        "countdown": 0.0,
        "intrusion_count": 0,
        "any_intrusion": False,
        "lux_level": 0.0,
        "fans_active": False,
        "lights_on": False,
        "relay_cut_off": False,
        "kwh_saved": 0.0,
        "energy_status": "NORMAL",
        "active_alerts": []
    }
    if cam_id not in status_queues:
        return {"error": "Camera not found"}
    sq = status_queues[cam_id]
    try:
        status = None
        while not sq.empty():
            status = sq.get_nowait()
        if status:
            cached_camera_status[cam_id] = status
            status["occupancy_count"] = status.get("occupancy", 0)
            return status
        if cam_id in cached_camera_status:
            c = dict(cached_camera_status[cam_id])
            c["occupancy_count"] = c.get("occupancy", 0)
            return c
        return default_status
    except:
        return default_status

# --- Energy & Security Analytics Endpoints (Phase 3/4) ---
@app.get("/api/energy/summary")
async def api_get_energy_summary():
    from core.event_manager import event_manager
    return event_manager.fetch_energy_summary()

@app.get("/api/energy/events")
async def api_get_energy_events(limit: int = 30):
    from core.event_manager import event_manager
    return event_manager.fetch_recent_energy_events(limit=limit)

@app.get("/api/security/events")
async def api_get_security_events(limit: int = 30):
    from core.event_manager import event_manager
    return event_manager.fetch_recent_security_events(limit=limit)

@app.get("/api/clips")
async def api_list_clips():
    clips_dir = os.path.join("storage", "clips")
    os.makedirs(clips_dir, exist_ok=True)
    files = [f for f in os.listdir(clips_dir) if f.endswith(".mp4") or f.endswith(".webm")]
    # sort by creation time descending
    files.sort(key=lambda x: os.path.getmtime(os.path.join(clips_dir, x)), reverse=True)
    return [{"filename": f, "url": f"/clips/{f}"} for f in files]

from fastapi.responses import FileResponse
@app.get("/clips/{filename}")
async def get_clip(filename: str):
    file_path = os.path.join("storage", "clips", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404)
    media_type = "video/webm" if filename.endswith(".webm") else "video/mp4"
    return FileResponse(file_path, media_type=media_type)

# --- Video Streaming ---
def encode_frame(frame):
    ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return ret, buffer

async def generate_video_stream(cam_id):
    while True:
        fq = frame_queues.get(cam_id)
        if fq and not fq.empty():
            try:
                frame = fq.get_nowait()
                # Run CPU-bound imencode in thread pool to prevent blocking asyncio loop
                ret, buffer = await asyncio.to_thread(encode_frame, frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except:
                pass
        await asyncio.sleep(0.01) # Non-blocking async sleep to reduce lag and CPU spin

@app.get("/video_feed/{cam_id}")
async def video_feed(cam_id: str):
    return StreamingResponse(generate_video_stream(cam_id), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        if websocket in active_connections:
            active_connections.remove(websocket)

if __name__ == "__main__":
    mp.freeze_support() 
    uvicorn.run("run:app", host="0.0.0.0", port=8000, reload=False)
