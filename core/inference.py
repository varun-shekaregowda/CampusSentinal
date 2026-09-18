import cv2
import time
import numpy as np
import collections
import os
import multiprocessing as mp
from ultralytics import YOLO
from core.event_manager import event_manager
from core.config_manager import config_manager
import math

class CameraPipeline:
    def __init__(self, cam_config, alert_queue, command_queue=None, status_queue=None):
        self.cam_id = cam_config['id']
        url_val = cam_config['url']
        if url_val.isdigit():
            self.url = int(url_val)
        else:
            self.url = url_val
        self.mode = cam_config['mode']
        self.roi = cam_config['roi']
        self.alert_queue = alert_queue
        self.command_queue = command_queue
        
        print(f"[{self.cam_id}] Initializing YOLO models...")
        # For CLASSROOM we only need object detection (tracking)
        # For EXAM we need pose
        if self.mode == "EXAM":
            self.model = YOLO('yolov8n-pose.pt')
        else:
            self.model = YOLO('yolov8n.pt')
            
        self.cap = cv2.VideoCapture(self.url)
        # Removed CAP_PROP_BUFFERSIZE to prevent camera reading bottlenecks
        
        # State tracking
        self.suspect_registry = {}
        self.head_turn_buffers = {}
        self.FRAME_THRESHOLD = 30 # Approx 1.5-2 seconds at 15fps
        
        # Playback and Recording State
        self.status_queue = status_queue
        self.paused = False
        self.speed_factor = 1.0
        self.frame_buffer = collections.deque(maxlen=900) # Past 1 minute (approx 15fps * 60)
        self.is_capturing = False
        self.capture_frames_left = 0
        self.capture_writer = None
        self.capture_filename = ""
        self.loop_counter = 0

        # Multithreading state
        import queue
        self.latest_detections = {"boxes": [], "occupancy": 0}
        self.current_frame = None
        self.inference_running = True
        self.internal_queue = queue.Queue()

    def is_in_roi(self, box):
        if not self.roi:
            return False
        # Calculate bottom center of the bounding box (feet)
        x1, y1, x2, y2 = box
        feet_point = ((x1 + x2) / 2, y2)
        
        # Check if point is inside polygon using cv2
        poly = np.array(self.roi, np.int32)
        result = cv2.pointPolygonTest(poly, feet_point, False)
        return result >= 0

    def process_classroom(self, frame):
        # BoT-SORT/ByteTrack for occlusion/overlap
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, classes=[0], conf=0.25)[0]
        detections = {"boxes": [], "occupancy": len(results.boxes)}
        
        if results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                
                # Check ROI Intrusion
                if self.is_in_roi(box):
                    if track_id not in self.suspect_registry:
                        self.suspect_registry[track_id] = "ROI INTRUSION"
                        # Fire alert
                        alert = event_manager.log_event(self.cam_id, "ROI INTRUSION", f"Person {track_id} entered restricted zone.", frame)
                        self.alert_queue.put(alert)
                        
                color = (0, 0, 255) if track_id in self.suspect_registry else (0, 255, 0)
                detections["boxes"].append({
                    "coords": (x1, y1, x2, y2),
                    "color": color,
                    "text": f"ID:{track_id}"
                })
                
        return detections

    def analyze_pose_anomaly(self, kpts):
        """
        Analyzes 17-point YOLO pose keypoints to detect malpractice (turning around, looking sideways).
        Handles edge cases where students are not facing the camera perfectly.
        """
        nose, l_eye, r_eye, l_ear, r_ear, l_sh, r_sh = kpts[0:7]
        
        # 1. Severe Profile View (Turned 90 degrees left or right to look at neighbor)
        # If one side of the face is highly visible and the other is completely occluded.
        left_visible = l_ear[2] > 0.5 and l_eye[2] > 0.5
        right_visible = r_ear[2] > 0.5 and r_eye[2] > 0.5
        if (left_visible and r_ear[2] < 0.2 and r_eye[2] < 0.2) or \
           (right_visible and l_ear[2] < 0.2 and l_eye[2] < 0.2):
            return True

        # 2. Facing Backwards (Turned 180 degrees to look behind)
        # Shoulders and ears might be visible, but nose and eyes are completely hidden.
        if nose[2] < 0.15 and l_eye[2] < 0.15 and r_eye[2] < 0.15:
            # Confirm we actually see the person's upper body to avoid false positives
            if l_sh[2] > 0.5 and r_sh[2] > 0.5:
                return True

        # 3. Moderate Head Turn (45-80 degrees) using geometric ratio of the nose between ears
        if l_ear[2] > 0.4 and r_ear[2] > 0.4 and nose[2] > 0.5:
            ear_dist = abs(l_ear[0] - r_ear[0])
            if ear_dist > 15: # Prevent division by zero
                nose_ratio = abs(nose[0] - l_ear[0]) / ear_dist
                yaw = (nose_ratio - 0.5) * 180
                if abs(yaw) > 45:
                    return True
                    
        # If looking down at the paper, eyes/nose might drop in confidence, 
        # but unless it triggers the above, they are considered safe.
        return False

    def process_exam(self, frame):
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, classes=[0], conf=0.25)[0]
        detections = {"boxes": [], "occupancy": len(results.boxes)}
        
        if results.boxes.id is not None and results.keypoints is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            keypoints_data = results.keypoints.data.cpu().numpy()
            
            for box, track_id, kpts in zip(boxes, track_ids, keypoints_data):
                x1, y1, x2, y2 = map(int, box)
                
                # Analyze pose for cheating anomalies
                is_anomaly = self.analyze_pose_anomaly(kpts)
                    
                if is_anomaly:
                    self.head_turn_buffers[track_id] = self.head_turn_buffers.get(track_id, 0) + 1
                    if self.head_turn_buffers[track_id] >= self.FRAME_THRESHOLD and track_id not in self.suspect_registry:
                        self.suspect_registry[track_id] = "SUSTAINED HEAD TURN"
                        # Signal the Output Thread to do the 15-second auto capture
                        self.internal_queue.put({
                            "action": "auto_capture_15s",
                            "track_id": track_id,
                            "description": f"Student {track_id} looking away for >2s.",
                            "frame": frame.copy()
                        })
                else:
                    self.head_turn_buffers[track_id] = max(0, self.head_turn_buffers.get(track_id, 0) - 2)
                    
                if track_id in self.suspect_registry:
                    color = (0, 0, 255) # Red
                elif self.head_turn_buffers.get(track_id, 0) > 0:
                    color = (0, 255, 255) # Yellow
                else:
                    color = (0, 255, 0) # Green
                    
                detections["boxes"].append({
                    "coords": (x1, y1, x2, y2),
                    "color": color,
                    "text": f"ID:{track_id}"
                })
                
        return detections

    def inference_loop(self):
        """Runs AI tracking entirely in the background"""
        while self.inference_running:
            if self.current_frame is None or self.paused:
                time.sleep(0.01)
                continue
                
            frame = self.current_frame.copy()
            if self.mode == "EXAM":
                detections = self.process_exam(frame)
            else:
                detections = self.process_classroom(frame)
                
            self.latest_detections = detections
            # Slight yield to avoid 100% core pinning constantly
            time.sleep(0.001)

    def run(self, frame_queue):
        """Runs continuously in a separate process, outputs annotated frames to a queue for the UI."""
        import threading
        
        # Calculate natural FPS for throttling
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or math.isnan(fps): fps = 30.0
        frame_time = 1.0 / fps
        is_file = isinstance(self.url, str) and not (self.url.startswith("http") or self.url.startswith("rtsp") or self.url.isdigit())
        
        # Start AI inference thread
        t = threading.Thread(target=self.inference_loop)
        t.daemon = True
        t.start()
        
        while self.cap.isOpened() and not getattr(self.command_queue, 'shutdown_flag', False):
            start_t = time.time()
            
            if self.command_queue and not self.command_queue.empty():
                try:
                    cmd = self.command_queue.get_nowait()
                    action = cmd.get("action")
                    if action == "clear":
                        tid = cmd.get("track_id")
                        if tid in self.suspect_registry:
                            del self.suspect_registry[tid]
                        if tid in self.head_turn_buffers:
                            self.head_turn_buffers[tid] = 0
                    elif action == "pause":
                        self.paused = True
                    elif action == "play":
                        self.paused = False
                    elif action == "seek":
                        if isinstance(self.url, str) and not self.url.isdigit():
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, cmd.get("value"))
                    elif action == "speed":
                        self.speed_factor = cmd.get("value")
                    elif action == "capture":
                        if not self.is_capturing:
                            self.is_capturing = True
                            self.capture_frames_left = 900
                            self.capture_filename = os.path.join("storage", "clips", f"clip_{self.cam_id}_{int(time.time())}.mp4")
                            os.makedirs(os.path.dirname(self.capture_filename), exist_ok=True)
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            w_fps = self.cap.get(cv2.CAP_PROP_FPS)
                            if w_fps == 0 or math.isnan(w_fps): w_fps = 15.0
                            self.capture_writer = cv2.VideoWriter(self.capture_filename, fourcc, w_fps, (1280, 720))
                            # Dump history buffer
                            for b_frame in list(self.frame_buffer):
                                self.capture_writer.write(b_frame)
                except Exception as e:
                    print(e)
                    
            if hasattr(self, 'internal_queue') and not self.internal_queue.empty():
                try:
                    cmd = self.internal_queue.get_nowait()
                    if cmd["action"] == "auto_capture_15s":
                        # Pull up to 450 frames (15s at 30fps)
                        clip_frames = list(self.frame_buffer)[-450:]
                        
                        def write_incident(frames, frame_img, track_id, desc):
                            ts = int(time.time())
                            vid_filename = os.path.join("storage", "clips", f"auto_{self.cam_id}_{track_id}_{ts}.mp4")
                            os.makedirs(os.path.dirname(vid_filename), exist_ok=True)
                            
                            w_fps = self.cap.get(cv2.CAP_PROP_FPS)
                            if w_fps == 0 or math.isnan(w_fps): w_fps = 30.0
                            
                            writer = cv2.VideoWriter(vid_filename, cv2.VideoWriter_fourcc(*'mp4v'), w_fps, (1280, 720))
                            for f in frames:
                                writer.write(f)
                            writer.release()
                            
                            # Log to DB and emit alert to UI
                            alert = event_manager.log_event(self.cam_id, "MALPRACTICE", desc, frame=frame_img, video_path=vid_filename)
                            self.alert_queue.put(alert)
                            
                        t2 = threading.Thread(target=write_incident, args=(clip_frames, cmd["frame"], cmd["track_id"], cmd["description"]))
                        t2.daemon = True
                        t2.start()
                except Exception as e:
                    print("Internal queue error:", e)

            if self.status_queue:
                self.loop_counter += 1
                if self.loop_counter % 5 == 0: # Update every 5 loops
                    frame_idx = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                    total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    try:
                        while not self.status_queue.empty():
                            self.status_queue.get_nowait()
                        self.status_queue.put({"frame": frame_idx, "total": total_frames, "paused": self.paused, "speed": self.speed_factor})
                    except: pass
            
            if self.paused:
                time.sleep(0.1)
                continue

            if self.speed_factor > 1.0:
                skip = int(self.speed_factor) - 1
                for _ in range(skip):
                    self.cap.grab()

            ret, frame = self.cap.read()
            if not ret:
                # Loop video for testing purposes if it ends
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            # Resize for performance and UI fitting
            frame = cv2.resize(frame, (1280, 720))
            self.current_frame = frame.copy()
            self.frame_buffer.append(frame.copy())
            
            if self.is_capturing:
                if self.capture_writer:
                    self.capture_writer.write(frame)
                self.capture_frames_left -= 1
                if self.capture_frames_left <= 0:
                    self.is_capturing = False
                    if self.capture_writer:
                        self.capture_writer.release()
                        self.capture_writer = None
                    alert = event_manager.log_event(self.cam_id, "VIDEO_CAPTURE", f"Saved 2-minute clip to {self.capture_filename}")
                    self.alert_queue.put(alert)
            
            # ---------------------------------------------
            # DRAW LATEST DETECTIONS (Zero Latency)
            # ---------------------------------------------
            dets = self.latest_detections
            
            # Draw ROI if applicable
            if self.roi and self.mode == "CLASSROOM":
                cv2.polylines(frame, [np.array(self.roi, np.int32)], True, (255, 0, 0), 2)
                
            for b in dets.get("boxes", []):
                x1, y1, x2, y2 = b["coords"]
                color = b["color"]
                text = b["text"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
            cv2.putText(frame, f"OCCUPANCY: {dets.get('occupancy', 0)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
            # Keep the queue small so it doesn't build up latency
            if frame_queue.full():
                try:
                    frame_queue.get_nowait() # Drop oldest frame
                except:
                    pass
            frame_queue.put(frame)
            
            # Regulate playback speed for local video files so they play at correct FPS
            if is_file:
                elapsed = time.time() - start_t
                sleep_time = (frame_time / self.speed_factor) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            
        self.inference_running = False
        self.cap.release()

def run_camera_process(cam_config, frame_queue, alert_queue, command_queue=None, status_queue=None):
    pipeline = CameraPipeline(cam_config, alert_queue, command_queue, status_queue)
    pipeline.run(frame_queue)
