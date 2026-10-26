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

class ExamMalpracticeAnalyzer:
    """
    Evaluates 17-keypoint skeleton outputs from YOLOv8-Pose.
    Keypoint indices:
      0: Nose, 1: Left Eye, 2: Right Eye, 3: Left Ear, 4: Right Ear,
      5: Left Shoulder, 6: Right Shoulder, 9: Left Wrist, 10: Right Wrist
    """
    def __init__(self, fps=25, trigger_sec=2.5):
        self.fps = fps
        self.trigger_threshold = int(fps * trigger_sec)
        self.student_trackers = {} # track_id -> suspicion_counter
        self.student_reasons = {}  # track_id -> list of reasons

    def analyze_student(self, track_id: int, keypoints: np.ndarray, confidences: np.ndarray, box=None) -> dict:
        if track_id not in self.student_trackers:
            self.student_trackers[track_id] = 0

        # Standing vs. Sitting Filter (Teacher / Invigilator Detection)
        if box is not None:
            x1, y1, x2, y2 = box
            width = max(1.0, float(x2 - x1))
            height = max(1.0, float(y2 - y1))
            if height > 1.7 * width and height > 130:
                self.student_trackers[track_id] = 0
                self.student_reasons.pop(track_id, None)
                return {
                    "track_id": track_id,
                    "suspicion_level": 0.0,
                    "is_malpractice": False,
                    "is_invigilator": True,
                    "reasons": []
                }

        nose_conf = confidences[0]
        l_ear_conf, r_ear_conf = confidences[3], confidences[4]
        l_sh_conf, r_sh_conf = confidences[5], confidences[6]

        is_suspicious = False
        reasons = []

        if l_sh_conf > 0.4 and r_sh_conf > 0.4:
            l_sh = keypoints[5]
            r_sh = keypoints[6]
            shoulder_width = np.linalg.norm(l_sh - r_sh) + 1e-6

            # 180-Degree Turn (Facing Backwards)
            if nose_conf < 0.2 and l_ear_conf < 0.2 and r_ear_conf < 0.2:
                is_suspicious = True
                reasons.append("Facing Backwards (180 Degree Turn)")
            elif nose_conf > 0.4:
                nose_x = keypoints[0][0]
                if (l_ear_conf < 0.25 and r_ear_conf > 0.45) or (r_ear_conf < 0.25 and l_ear_conf > 0.45):
                    is_suspicious = True
                    reasons.append("Lateral Head Turn")
                elif l_ear_conf > 0.35 and r_ear_conf > 0.35:
                    d_left = abs(nose_x - keypoints[3][0])
                    d_right = abs(nose_x - keypoints[4][0])
                    yaw_asymmetry = abs(d_left - d_right) / shoulder_width
                    if yaw_asymmetry > 0.65:
                        is_suspicious = True
                        reasons.append("Sustained Head Yaw")

            for wrist_idx, side in [(9, "Left"), (10, "Right")]:
                if confidences[wrist_idx] > 0.4:
                    wrist_x = keypoints[wrist_idx][0]
                    if wrist_x < (min(l_sh[0], r_sh[0]) - 0.85 * shoulder_width) or \
                       wrist_x > (max(l_sh[0], r_sh[0]) + 0.85 * shoulder_width):
                        is_suspicious = True
                        reasons.append(f"{side} Arm Extension")

        if is_suspicious:
            self.student_trackers[track_id] += 1
            if reasons:
                self.student_reasons[track_id] = reasons
        else:
            self.student_trackers[track_id] = max(0, self.student_trackers[track_id] - 2)
            if self.student_trackers[track_id] == 0:
                self.student_reasons.pop(track_id, None)

        confirmed = self.student_trackers[track_id] >= self.trigger_threshold
        current_reasons = reasons if reasons else self.student_reasons.get(track_id, [])

        return {
            "track_id": track_id,
            "suspicion_level": min(1.0, self.student_trackers[track_id] / self.trigger_threshold),
            "is_malpractice": confirmed,
            "is_invigilator": False,
            "reasons": current_reasons
        }

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
        
        # Determine natural FPS for analyzer
        fps_val = self.cap.get(cv2.CAP_PROP_FPS)
        if fps_val <= 0 or math.isnan(fps_val):
            fps_val = 25.0
        self.exam_analyzer = ExamMalpracticeAnalyzer(fps=int(fps_val), trigger_sec=2.5)
        
        # State tracking
        self.suspect_registry = {}
        self.head_turn_buffers = {}
        self.FRAME_THRESHOLD = 30 # Approx 1.5-2 seconds at 15fps
        
        # Playback and Recording State
        self.status_queue = status_queue
        self.paused = False
        self.speed_factor = 1.0
        self.frame_buffer = collections.deque(maxlen=900) # 30 seconds at ~30 FPS
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

    def process_exam(self, frame):
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, classes=[0], conf=0.15, imgsz=736)[0]
        detections = {"boxes": [], "occupancy": len(results.boxes)}
        new_incidents = []
        
        if results.boxes.id is not None and results.keypoints is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            keypoints_xy = results.keypoints.xy.cpu().numpy()
            confidences = results.keypoints.conf.cpu().numpy() if results.keypoints.conf is not None else results.keypoints.data.cpu().numpy()[:, :, 2]
            
            for box, track_id, kpt, conf in zip(boxes, track_ids, keypoints_xy, confidences):
                x1, y1, x2, y2 = map(int, box)
                
                # Analyze student pose using ExamMalpracticeAnalyzer with bounding box aspect ratio
                analysis = self.exam_analyzer.analyze_student(track_id, kpt, conf, box=(x1, y1, x2, y2))
                
                if analysis.get("is_invigilator"):
                    # Invigilator: Ignore for malpractice, completely skip drawing
                    if track_id in self.suspect_registry:
                        del self.suspect_registry[track_id]
                elif analysis["is_malpractice"] or track_id in self.suspect_registry:
                    color = (0, 0, 255) # Red (0, 0, 255): Confirmed Malpractice
                    reasons_list = analysis.get("reasons", [])
                    reasons_str = ", ".join(reasons_list) if reasons_list else self.suspect_registry.get(track_id, "Malpractice")
                    label = f"ID: {track_id} | {reasons_str}"
                    
                    if track_id not in self.suspect_registry:
                        self.suspect_registry[track_id] = reasons_str
                        new_incidents.append({
                            "track_id": track_id,
                            "description": f"Student {track_id}: {reasons_str}"
                        })
                    detections["boxes"].append({
                        "coords": (x1, y1, x2, y2),
                        "color": color,
                        "text": label
                    })
                elif analysis["suspicion_level"] > 0:
                    color = (0, 255, 255) # Yellow (0, 255, 255): Suspicion Building
                    suspicion_pct = int(analysis["suspicion_level"] * 100)
                    reasons_list = analysis.get("reasons", [])
                    reasons_str = ", ".join(reasons_list) if reasons_list else ""
                    if reasons_str:
                        label = f"ID: {track_id} ({suspicion_pct}%) | {reasons_str}"
                    else:
                        label = f"ID: {track_id} ({suspicion_pct}%)"
                    detections["boxes"].append({
                        "coords": (x1, y1, x2, y2),
                        "color": color,
                        "text": label
                    })
                else:
                    # Normal: completely skip drawing green bounding boxes
                    pass
                
        # Draw visual telemetry ONLY for Red (Malpractice) or Yellow (Suspicion Building)
        for b in detections["boxes"]:
            bx1, by1, bx2, by2 = b["coords"]
            bcolor = b["color"]
            btext = b["text"]
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), bcolor, 2)
            cv2.putText(frame, btext, (bx1, max(20, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bcolor, 2)
            # 17 skeleton keypoints rendering removed to save CPU cycles and clean up video
                    
        cv2.putText(frame, f"OCCUPANCY: {detections.get('occupancy', 0)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Append annotated frame (with burned visual telemetry) to rolling ring buffer for evidence clips
        self.frame_buffer.append(frame.copy())
        
        # Signal output thread for any new incidents with high-res annotated frame
        for inc in new_incidents:
            self.internal_queue.put({
                "action": "auto_capture_30s",
                "track_id": inc["track_id"],
                "description": inc["description"],
                "frame": frame.copy()
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

    def stop_and_finalize_capture(self, current_frame=None):
        """Safely flushes and closes any active manual capture VideoWriter, writing the MP4 moov atom."""
        if self.is_capturing or self.capture_writer is not None:
            self.is_capturing = False
            writer = self.capture_writer
            self.capture_writer = None
            if writer:
                try:
                    writer.release()
                except Exception as e:
                    print("[CampusSentinel] Error releasing capture writer:", e)
            if self.capture_filename:
                db_vid_path = self.capture_filename.replace('\\', '/')
                thumb = current_frame if current_frame is not None else self.current_frame
                alert = event_manager.log_event(
                    self.cam_id, 
                    "VIDEO_CAPTURE", 
                    f"Saved manual clip to {db_vid_path}", 
                    frame=thumb, 
                    video_path=db_vid_path
                )
                if hasattr(self, 'alert_queue') and self.alert_queue:
                    try:
                        self.alert_queue.put(alert)
                    except:
                        pass
                print(f"[CampusSentinel] Finalized manual recording: {db_vid_path}")
                self.capture_filename = ""

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
                        if hasattr(self, 'exam_analyzer'):
                            if tid in self.exam_analyzer.student_trackers:
                                self.exam_analyzer.student_trackers[tid] = 0
                            if hasattr(self.exam_analyzer, 'student_reasons') and tid in self.exam_analyzer.student_reasons:
                                del self.exam_analyzer.student_reasons[tid]
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
                        if self.is_capturing:
                            # User clicked again: stop recording early and finalize immediately
                            self.stop_and_finalize_capture(current_frame=self.current_frame)
                        else:
                            self.is_capturing = True
                            w_fps = self.cap.get(cv2.CAP_PROP_FPS)
                            if w_fps <= 0 or math.isnan(w_fps): w_fps = 30.0
                            
                            ts = int(time.time())
                            clip_mp4 = f"storage/clips/clip_{self.cam_id}_{ts}.mp4"
                            clip_webm = f"storage/clips/clip_{self.cam_id}_{ts}.webm"
                            os.makedirs("storage/clips", exist_ok=True)
                            
                            fourcc = cv2.VideoWriter_fourcc(*'avc1')
                            self.capture_writer = cv2.VideoWriter(clip_mp4, fourcc, w_fps, (1280, 720))
                            self.capture_filename = clip_mp4
                            if not self.capture_writer.isOpened():
                                fourcc = cv2.VideoWriter_fourcc(*'vp09')
                                self.capture_writer = cv2.VideoWriter(clip_webm, fourcc, w_fps, (1280, 720))
                                self.capture_filename = clip_webm
                                
                            self.capture_filename = self.capture_filename.replace('\\', '/')
                            
                            # Prepend recent buffer history (up to 300 frames ~15s to keep write instant without stutter)
                            buf = list(self.frame_buffer)
                            recent_buf = buf[-300:] if len(buf) > 300 else buf
                            for b_frame in recent_buf:
                                self.capture_writer.write(b_frame)
                                
                            # 2 minutes = 120 seconds * w_fps (e.g. 2400 frames at 20fps)
                            total_target = int(120 * w_fps)
                            remaining = max(total_target - len(recent_buf), int(30 * w_fps))
                            self.capture_frames_left = remaining
                            self.capture_total_frames = remaining
                            print(f"[CampusSentinel] Started manual recording: {self.capture_filename} ({remaining} frames remaining)")
                    elif action == "stop_capture":
                        self.stop_and_finalize_capture(current_frame=self.current_frame)
                    elif action == "shutdown":
                        break
                except Exception as e:
                    print(e)
                    
            if hasattr(self, 'internal_queue') and not self.internal_queue.empty():
                try:
                    cmd = self.internal_queue.get_nowait()
                    if cmd["action"] in ("auto_capture_30s", "auto_capture_15s"):
                        # Pull full 30 seconds from buffer (up to 900 frames at ~30 FPS)
                        clip_frames = list(self.frame_buffer)[-900:]
                        
                        def write_incident(frames, frame_img, track_id, desc):
                            ts = int(time.time())
                            vid_filename_mp4 = f"storage/clips/auto_{self.cam_id}_{track_id}_{ts}.mp4"
                            vid_filename_webm = f"storage/clips/auto_{self.cam_id}_{track_id}_{ts}.webm"
                            os.makedirs("storage/clips", exist_ok=True)
                            
                            w_fps = self.cap.get(cv2.CAP_PROP_FPS)
                            if w_fps <= 0 or math.isnan(w_fps): w_fps = 30.0
                            
                            # CRITICAL CODEC FIX: Try avc1 first with .mp4 for HTML5 web browser playback, fallback to vp09 with .webm
                            fourcc = cv2.VideoWriter_fourcc(*'avc1')
                            writer = cv2.VideoWriter(vid_filename_mp4, fourcc, w_fps, (1280, 720))
                            vid_filename = vid_filename_mp4
                            
                            if not writer.isOpened():
                                fourcc = cv2.VideoWriter_fourcc(*'vp09')
                                writer = cv2.VideoWriter(vid_filename_webm, fourcc, w_fps, (1280, 720))
                                vid_filename = vid_filename_webm
                                
                            for f in frames:
                                writer.write(f)
                            writer.release()
                            
                            # Ensure file paths saved to the database use forward slashes (/)
                            db_vid_path = vid_filename.replace('\\', '/')
                            alert = event_manager.log_event(self.cam_id, "MALPRACTICE", desc, frame=frame_img, video_path=db_vid_path)
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
                    w_fps = self.cap.get(cv2.CAP_PROP_FPS)
                    if w_fps <= 0 or math.isnan(w_fps): w_fps = 20.0
                    try:
                        while not self.status_queue.empty():
                            self.status_queue.get_nowait()
                        self.status_queue.put({
                            "frame": frame_idx, 
                            "total": total_frames, 
                            "paused": self.paused, 
                            "speed": self.speed_factor,
                            "is_capturing": self.is_capturing,
                            "capture_left": max(0, self.capture_frames_left),
                            "capture_total": getattr(self, 'capture_total_frames', 2400),
                            "fps": int(w_fps)
                        })
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
                cv2.putText(frame, text, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                
            cv2.putText(frame, f"OCCUPANCY: {dets.get('occupancy', 0)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if self.mode != "EXAM":
                # For non-exam modes, append annotated frame here
                self.frame_buffer.append(frame.copy())
            
            if self.is_capturing:
                if self.capture_writer:
                    self.capture_writer.write(frame)
                self.capture_frames_left -= 1
                if self.capture_frames_left <= 0:
                    self.stop_and_finalize_capture(current_frame=frame)
                
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
            
        self.stop_and_finalize_capture()
        self.inference_running = False
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

def run_camera_process(cam_config, frame_queue, alert_queue, command_queue=None, status_queue=None):
    pipeline = CameraPipeline(cam_config, alert_queue, command_queue, status_queue)
    pipeline.run(frame_queue)
