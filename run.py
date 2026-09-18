import multiprocessing as mp
import uvicorn
from fastapi import FastAPI, WebSocket, Request, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List, Tuple
import cv2
import asyncio
import json
import time
import os
import shutil

from core.config_manager import config_manager
from core.inference import run_camera_process

app = FastAPI()
app.mount("/static", StaticFiles(directory="ui/static"), name="static")
app.mount("/screenshots", StaticFiles(directory="storage/screenshots"), name="screenshots")
app.mount("/clips", StaticFiles(directory="storage/clips"), name="clips")
templates = Jinja2Templates(directory="ui/templates")

frame_queues = {}
command_queues = {}
status_queues = {}
active_processes = {}
alert_queue = mp.Queue()
active_connections = []

class CameraCreate(BaseModel):
    name: str
    url: str
    mode: str
    roi: Optional[List[Tuple[int, int]]] = None

class CameraUpdate(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    mode: Optional[str] = None
    roi: Optional[List[Tuple[int, int]]] = None

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
        active_processes[cam_id].terminate()
        active_processes[cam_id].join()
        del active_processes[cam_id]
        print(f"Stopped Inference Process for {cam_id}")
    if cam_id in frame_queues:
        del frame_queues[cam_id]
    if cam_id in command_queues:
        del command_queues[cam_id]
    if cam_id in status_queues:
        del status_queues[cam_id]

@app.on_event("startup")
async def startup_event():
    for cam in config_manager.get_cameras():
        start_camera_process(cam)
    asyncio.create_task(broadcast_alerts())

async def broadcast_alerts():
    while True:
        if not alert_queue.empty():
            alert = alert_queue.get_nowait()
            message = json.dumps(alert)
            for connection in list(active_connections):
                try:
                    await connection.send_text(message)
                except:
                    if connection in active_connections:
                        active_connections.remove(connection)
        await asyncio.sleep(0.1)

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
    return [dict(ix) for ix in rows]

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
    start_camera_process(updated_cam)
    return updated_cam

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
    if cam_id not in status_queues:
        return {"error": "Camera not found"}
    sq = status_queues[cam_id]
    if sq.empty():
        return {"frame": 0, "total": 0, "paused": False, "speed": 1.0}
    try:
        # Get the latest status in queue
        status = None
        while not sq.empty():
            status = sq.get_nowait()
        return status
    except:
        return {"frame": 0, "total": 0, "paused": False, "speed": 1.0}

@app.get("/api/clips")
async def api_list_clips():
    clips_dir = os.path.join("storage", "clips")
    os.makedirs(clips_dir, exist_ok=True)
    files = [f for f in os.listdir(clips_dir) if f.endswith(".mp4")]
    # sort by creation time descending
    files.sort(key=lambda x: os.path.getmtime(os.path.join(clips_dir, x)), reverse=True)
    return [{"filename": f, "url": f"/clips/{f}"} for f in files]

from fastapi.responses import FileResponse
@app.get("/clips/{filename}")
async def get_clip(filename: str):
    file_path = os.path.join("storage", "clips", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404)
    return FileResponse(file_path, media_type="video/mp4")

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
