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

class MotionGator:
    """
    Edge CPU optimization layer using MOG2 Background Subtraction.
    Gating ignores static frames to save CPU cycles, while a periodic
    heartbeat ensures stationary occupants are never missed.
    """
    def __init__(self, min_area=800, heartbeat=25):
        self.min_area = min_area
        self.heartbeat = heartbeat
        self.frame_counter = 0
        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=50,
            history=500
        )
        self.kernel = np.ones((5, 5), np.uint8)

    def check_motion(self, frame) -> tuple:
        self.frame_counter += 1
        is_heartbeat = (self.frame_counter % self.heartbeat == 0)

        fg_mask = self.back_sub.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_detected = False
        motion_boxes = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_area:
                motion_detected = True
                x, y, w, h = cv2.boundingRect(contour)
                motion_boxes.append((x, y, w, h))

        should_run_yolo = motion_detected or is_heartbeat
        return should_run_yolo, fg_mask, motion_boxes

    def reset(self):
        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=50,
            history=500
        )

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

    def analyze_student(self, track_id: int, keypoints: np.ndarray, confidences: np.ndarray, box=None, camera_angle: str = "REAR") -> dict:
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
        l_wrist_conf, r_wrist_conf = confidences[9], confidences[10]

        is_suspicious = False
        reasons = []

        if camera_angle == "FRONT":
            # ----------------------------------------------------
            # FRONT-FACING HEURISTICS (Lap/Phone & Under-Desk Hands)
            # ----------------------------------------------------
            # 1. Looking Down (Lap/Phone Check):
            # If the nose and shoulders are detected, calculate vertical distance between nose and shoulder line.
            if (l_sh_conf > 0.3 or r_sh_conf > 0.3) and nose_conf > 0.3:
                l_sh = keypoints[5]
                r_sh = keypoints[6]
                nose = keypoints[0]
                
                # Shoulder Y reference: use average if both visible, or the visible shoulder
                if l_sh_conf > 0.3 and r_sh_conf > 0.3:
                    shoulder_y = (l_sh[1] + r_sh[1]) / 2.0
                    shoulder_width = np.linalg.norm(l_sh - r_sh) + 1e-6
                elif l_sh_conf > 0.3:
                    shoulder_y = l_sh[1]
                    shoulder_width = 80.0
                else:
                    shoulder_y = r_sh[1]
                    shoulder_width = 80.0

                # In image coordinates, Y increases downward.
                # When looking up/forward, nose Y is significantly above shoulders: shoulder_y - nose[1] > 30px.
                # If nose drops abnormally low toward or below the shoulder line:
                vertical_dist = shoulder_y - nose[1]
                threshold = max(10.0, 0.15 * shoulder_width)
                if vertical_dist < threshold:
                    is_suspicious = True
                    reasons.append("Suspicious Head Pitch (Looking Down)")

            # 2. Hands Under Desk:
            # If shoulders are visible but both wrists vanish (l_wrist_conf < 0.15 and r_wrist_conf < 0.15)
            if (l_sh_conf > 0.3 or r_sh_conf > 0.3) and (l_wrist_conf < 0.15 and r_wrist_conf < 0.15):
                is_suspicious = True
                reasons.append("Hands Hidden Under Desk")

            # 3. Lateral Head Turn (Facing Left/Right from Front view)
            if nose_conf > 0.35:
                nose_x = keypoints[0][0]
                if (l_ear_conf < 0.2 and r_ear_conf > 0.45) or (r_ear_conf < 0.2 and l_ear_conf > 0.45):
                    is_suspicious = True
                    reasons.append("Lateral Head Turn")

        else:
            # ----------------------------------------------------
            # REAR-FACING HEURISTICS (Lateral Arm Extension & 180 Turn)
            # ----------------------------------------------------
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
        
        print(f"[{self.cam_id}] Initializing YOLO model for mode: {self.mode}...")
        # Load yolov8n-pose.pt for EXAM, EXAM_FRONT, EXAM_REAR, but lighter yolov8n.pt for SECURITY and ENERGY modes
        if self.mode in ("EXAM", "EXAM_FRONT", "EXAM_REAR"):
            self.model = YOLO('yolov8n-pose.pt')
        elif self.mode in ("SECURITY", "ENERGY", "CLASSROOM"):
            self.model = YOLO('yolov8n.pt')
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

        self.cam_name = cam_config.get('name', f"Camera-{self.cam_id}")

        # Multithreading state
        import queue
        self.latest_detections = {"boxes": [], "occupancy": 0}
        self.current_frame = None
        self.inference_running = True
        self.internal_queue = queue.Queue()

        # Motion gating layer (Edge CPU optimization)
        self.motion_gator = MotionGator(min_area=800, heartbeat=25)
        self.cached_detections = {"boxes": [], "occupancy": 0}

        # Module 2: Restricted Campus Zone Sentinel state
        self.track_history = {}          # track_id -> list of foot points [(x,y), ...]
        self.track_inside_zones = {}     # track_id -> bool
        self.track_entry_times = {}      # track_id -> float timestamp
        self.track_loiter_flagged = {}   # track_id -> bool
        self.track_last_alert_time = {}  # track_id -> float timestamp

        # Module 3: Smart Hall & Auditorium Energy Wastage Sentinel state
        self.prev_gray_frame = None
        self.empty_start_time = None
        self.energy_status = "NORMAL"    # "NORMAL", "COUNTDOWN", "ENERGY_WASTAGE", "ALL_DEVICES_OFF"
        self.relay_cut_off = False
        self.lights_on = False
        self.fans_rotating = False
        self.current_illumination = 0.0
        self.total_energy_saved_kwh = 0.0
        self.active_actions = []
        self.flash_state = False
        self.last_flash_time = time.time()

    def is_in_roi(self, box):
        if not self.roi:
            # In SECURITY mode, if no custom polygon ROI was defined, treat the entire camera feed as restricted zone
            if self.mode == "SECURITY":
                return True
            return False
        # Calculate bottom center of the bounding box (feet) and box center
        x1, y1, x2, y2 = box
        feet_point = ((x1 + x2) / 2.0, float(y2))
        center_point = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        
        # Check if point is inside polygon using cv2
        poly = np.array(self.roi, np.int32)
        if cv2.pointPolygonTest(poly, feet_point, False) >= 0:
            return True
        if cv2.pointPolygonTest(poly, center_point, False) >= 0:
            return True
        return False

    def _filter_false_alarm(self, box):
        """Filters out non-human artifacts (leaves, debris, wide shadows) based on area and human aspect ratio."""
        x1, y1, x2, y2 = box
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        area = w * h
        aspect_ratio = h / float(w)
        if area < 500 or area > 500000:
            return False
        if aspect_ratio < 0.6 or aspect_ratio > 5.5:
            return False
        return True

    def _compute_directional_entry(self, trajectory):
        """Checks if movement vector crosses from outside into the restricted polygon."""
        if not self.roi or len(trajectory) < 2:
            return False, (0, 0)
        poly = np.array(self.roi, np.int32)
        start_pt = trajectory[max(0, len(trajectory) - 5)]
        end_pt = trajectory[-1]
        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        disp = math.hypot(dx, dy)
        was_outside = cv2.pointPolygonTest(poly, (float(start_pt[0]), float(start_pt[1])), False) < 0
        now_inside = cv2.pointPolygonTest(poly, (float(end_pt[0]), float(end_pt[1])), False) >= 0
        return (was_outside and now_inside and disp >= 15), (int(dx), int(dy))

    def _calculate_illumination(self, frame: np.ndarray) -> float:
        """Calculates average frame luminance (Lux)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.current_illumination = float(np.mean(gray))
        return self.current_illumination

    def _detect_fan_rotation(self, frame: np.ndarray) -> bool:
        """Detects ceiling fan rotation in the top 35% ceiling ROI via frame difference."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        ceiling_h = int(h * 0.35)
        is_rotating = False
        if self.prev_gray_frame is not None and self.prev_gray_frame.shape == gray.shape:
            diff = cv2.absdiff(self.prev_gray_frame[:ceiling_h, :], gray[:ceiling_h, :])
            motion_ratio = float(np.mean(diff > 18))
            is_rotating = (motion_ratio > 0.005)
        self.prev_gray_frame = gray
        self.fans_rotating = is_rotating
        return is_rotating

    def process_classroom(self, frame):
        # Motion Gating Check (Edge CPU power saver)
        has_motion, _, _ = self.motion_gator.check_motion(frame)
        if not has_motion and self.cached_detections.get("boxes"):
            return self.cached_detections

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
        self.cached_detections = detections
        return detections

    def process_security(self, frame):
        """
        Module 2: Restricted Campus Zone Sentinel.
        - High-precision ByteTrack ID tracking
        - False alarm filtering (leaf/debris rejection via area and aspect ratio)
        - Point-in-polygon directional entry vectors
        - Dwell time loitering escalation
        - Instant intruder detection & photo evidence logging with 5-second temporal debounce
        """
        has_motion, _, _ = self.motion_gator.check_motion(frame)
        if not has_motion and not self.cached_detections.get("any_intrusion") and self.cached_detections.get("boxes"):
            return self.cached_detections

        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, classes=[0], conf=0.25)[0]
        valid_boxes = []
        valid_track_ids = []
        if results.boxes is not None and len(results.boxes) > 0:
            raw_boxes = results.boxes.xyxy.cpu().numpy()
            if results.boxes.id is not None:
                raw_track_ids = results.boxes.id.cpu().numpy().astype(int)
            else:
                raw_track_ids = np.arange(1, len(raw_boxes) + 1)
            for box, tid in zip(raw_boxes, raw_track_ids):
                if self._filter_false_alarm(box):
                    valid_boxes.append(box)
                    valid_track_ids.append(tid)
                else:
                    x1, y1, x2, y2 = box
                    if (x2 - x1) * (y2 - y1) >= 800:
                        valid_boxes.append(box)
                        valid_track_ids.append(tid)

        detections = {"boxes": [], "occupancy": len(valid_boxes), "any_intrusion": False, "active_alerts": []}
        now = time.time()

        for box, track_id in zip(valid_boxes, valid_track_ids):
            x1, y1, x2, y2 = map(int, box)
            foot_pt = (int((x1 + x2) / 2), y2)

            if track_id not in self.track_history:
                self.track_history[track_id] = []
            self.track_history[track_id].append(foot_pt)
            if len(self.track_history[track_id]) > 15:
                self.track_history[track_id].pop(0)

            inside = self.is_in_roi(box)
            if inside:
                # Explicitly set any_intrusion flag
                detections["any_intrusion"] = True
                alert_tag = f"ID:{track_id} INTRUSION"
                detections["active_alerts"].append(alert_tag)

                # Check directional entry
                is_dir, vec = self._compute_directional_entry(self.track_history[track_id])
                event_type = "DIRECTIONAL_ENTRY" if is_dir else "INTRUSION"
                detail = f"Vector ({vec[0]},{vec[1]}) into Restricted Zone" if is_dir else f"Intruder ID:{track_id} detected in Restricted Zone"

                # Check dwell time for loitering escalation
                if track_id not in self.track_entry_times:
                    self.track_entry_times[track_id] = now
                dwell = now - self.track_entry_times[track_id]
                if dwell > 3.0:
                    event_type = "LOITERING"
                    detail = f"Loitering in Restricted Zone for {dwell:.1f}s"

                # Temporal debounce: only trigger alert & photo snapshot once every 5 seconds per track ID
                last_alert = self.track_last_alert_time.get(track_id, 0.0)
                if now - last_alert >= 5.0:
                    self.track_last_alert_time[track_id] = now
                    alert = event_manager.log_security_event(
                        cam_id=self.cam_id,
                        event_type=event_type,
                        track_id=track_id,
                        zone_id="Zone-1",
                        zone_name=self.cam_name,
                        detail=detail,
                        frame=frame,
                        bbox=(x1, y1, x2, y2),
                        severity="HIGH" if event_type != "LOITERING" else "MEDIUM"
                    )
                    if self.alert_queue is not None:
                        try:
                            self.alert_queue.put(alert)
                        except Exception as e:
                            print(f"[SECURITY] Alert queue put error: {e}")

                color = (0, 0, 255) # Red (0, 0, 255) for intruder bounding box
                label = f"ID:{track_id} [INTRUDER]"
            else:
                if track_id in self.track_entry_times:
                    del self.track_entry_times[track_id]
                color = (0, 255, 0) # Green for clear
                label = f"ID:{track_id}"

            detections["boxes"].append({
                "coords": (x1, y1, x2, y2),
                "color": color,
                "text": label
            })

        detections["intrusion_count"] = sum(1 for b in detections["boxes"] if "[INTRUDER]" in b.get("text", ""))
        if detections["intrusion_count"] > 0:
            detections["any_intrusion"] = True
        self.cached_detections = detections
        return detections

    def process_energy(self, frame):
        """
        Module 3: Smart Hall & Auditorium Energy Wastage Sentinel.
        - Real-time Lux illumination monitoring
        - Upper 35% ceiling fan motion difference
        - 5-second Zero-Occupancy grace timer
        - Automated simulated relay power isolation
        """
        illumination = self._calculate_illumination(frame)
        self.lights_on = (illumination >= 65.0)
        self.fans_rotating = self._detect_fan_rotation(frame)

        has_motion, _, _ = self.motion_gator.check_motion(frame)

        if has_motion or not self.cached_detections.get("boxes"):
            results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, classes=[0], conf=0.25)[0]
            boxes = []
            if results.boxes.id is not None:
                raw_boxes = results.boxes.xyxy.cpu().numpy()
                raw_track_ids = results.boxes.id.cpu().numpy().astype(int)
                for box, track_id in zip(raw_boxes, raw_track_ids):
                    if self._filter_false_alarm(box):
                        x1, y1, x2, y2 = map(int, box)
                        boxes.append({
                            "coords": (x1, y1, x2, y2),
                            "color": (0, 255, 0),
                            "text": f"ID:{track_id}"
                        })
            self.cached_detections = {"boxes": boxes, "occupancy": len(boxes)}

        detections = dict(self.cached_detections)
        occupancy = detections.get("occupancy", 0)
        now = time.time()

        if occupancy > 0:
            self.empty_start_time = None
            self.energy_status = "NORMAL"
            self.relay_cut_off = False
            self.active_actions = []
        else:
            appliance_actions = []
            active_watts = 0.0
            if self.lights_on:
                appliance_actions.append("Turn OFF Lights (Lights are ON)")
                active_watts += 1200.0
            if self.fans_rotating:
                appliance_actions.append("Turn OFF Ceiling Fans (Fans are rotating)")
                active_watts += 300.0

            self.active_actions = appliance_actions

            if appliance_actions and not self.relay_cut_off:
                if self.empty_start_time is None:
                    self.empty_start_time = now

                empty_duration = now - self.empty_start_time
                if empty_duration >= 5.0:
                    self.energy_status = "ENERGY_WASTAGE"
                    self.relay_cut_off = True
                    kwh_saved = (active_watts * (empty_duration / 3600.0)) / 1000.0
                    self.total_energy_saved_kwh += kwh_saved
                    action_str = " & ".join(appliance_actions)
                    detail_msg = f"Zero occupancy in {self.cam_name}! Action: {action_str}"

                    alert = event_manager.log_energy_event(
                        cam_id=self.cam_id,
                        room_name=self.cam_name,
                        status="ENERGY_WASTAGE_RELAY_CUTOFF",
                        occupancy=0,
                        illumination=round(illumination, 1),
                        empty_duration=round(empty_duration, 1),
                        kwh_saved=round(kwh_saved, 4),
                        relay_triggered=True,
                        detail=detail_msg,
                        frame=frame
                    )
                    if self.alert_queue:
                        self.alert_queue.put(alert)
                else:
                    self.energy_status = "COUNTDOWN"
            elif not appliance_actions:
                self.energy_status = "ALL_DEVICES_OFF"
                self.empty_start_time = None

        countdown_rem = 0.0
        if self.empty_start_time is not None and not self.relay_cut_off:
            countdown_rem = max(0.0, round(5.0 - (now - self.empty_start_time), 1))

        detections["occupancy"] = occupancy
        detections["countdown"] = countdown_rem
        detections["lux_level"] = self.current_illumination
        detections["fans_active"] = self.fans_rotating
        detections["lights_on"] = self.lights_on
        detections["relay_cut_off"] = self.relay_cut_off
        detections["energy_status"] = self.energy_status
        detections["kwh_saved"] = self.total_energy_saved_kwh
        detections["active_actions"] = self.active_actions
        return detections

    def process_exam(self, frame):
        results = self.model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, classes=[0], conf=0.15, imgsz=736)[0]
        detections = {"boxes": [], "occupancy": len(results.boxes)}
        new_incidents = []
        
        # Determine camera angle based on pipeline mode
        camera_angle = "FRONT" if self.mode == "EXAM_FRONT" else "REAR"
        
        if results.boxes.id is not None and results.keypoints is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            track_ids = results.boxes.id.cpu().numpy().astype(int)
            keypoints_xy = results.keypoints.xy.cpu().numpy()
            confidences = results.keypoints.conf.cpu().numpy() if results.keypoints.conf is not None else results.keypoints.data.cpu().numpy()[:, :, 2]
            
            for box, track_id, kpt, conf in zip(boxes, track_ids, keypoints_xy, confidences):
                x1, y1, x2, y2 = map(int, box)
                
                # Analyze student pose using ExamMalpracticeAnalyzer with camera angle heuristic
                analysis = self.exam_analyzer.analyze_student(
                    track_id, kpt, conf, box=(x1, y1, x2, y2), camera_angle=camera_angle
                )
                
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
            if self.mode in ("EXAM", "EXAM_FRONT", "EXAM_REAR"):
                detections = self.process_exam(frame)
            elif self.mode == "SECURITY":
                detections = self.process_security(frame)
            elif self.mode == "ENERGY":
                detections = self.process_energy(frame)
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
                            "fps": int(w_fps),
                            # Extended Telemetry for Phase 3/4
                            "occupancy": self.latest_detections.get("occupancy", 0),
                            "countdown": self.latest_detections.get("countdown", 0.0),
                            "intrusion_count": self.latest_detections.get("intrusion_count", 0),
                            "any_intrusion": self.latest_detections.get("any_intrusion", False),
                            "lux_level": self.latest_detections.get("lux_level", 0.0),
                            "fans_active": self.latest_detections.get("fans_active", False),
                            "lights_on": self.latest_detections.get("lights_on", False),
                            "relay_cut_off": self.latest_detections.get("relay_cut_off", False),
                            "kwh_saved": self.latest_detections.get("kwh_saved", 0.0),
                            "energy_status": self.latest_detections.get("energy_status", "NORMAL"),
                            "active_alerts": self.latest_detections.get("active_alerts", [])
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

            # Toggle alert flashing banner
            if time.time() - self.last_flash_time > 0.4:
                self.flash_state = not self.flash_state
                self.last_flash_time = time.time()
            
            # ---------------------------------------------
            # DRAW LATEST DETECTIONS (Zero Latency)
            # ---------------------------------------------
            dets = self.latest_detections
            
            # 1. Draw ROI (CLASSROOM and SECURITY modes)
            if self.roi and self.mode in ("CLASSROOM", "SECURITY"):
                poly = np.array(self.roi, np.int32)
                poly_color = (0, 0, 255) if self.mode == "SECURITY" else (255, 0, 0)
                cv2.polylines(frame, [poly], True, poly_color, 2)
                if self.mode == "SECURITY":
                    overlay = frame.copy()
                    cv2.fillPoly(overlay, [poly], (0, 0, 255))
                    cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
                    rx, ry = self.roi[0]
                    cv2.rectangle(frame, (rx, max(0, ry - 22)), (rx + 230, ry), (10, 10, 10), -1)
                    cv2.putText(frame, "[RESTRICTED PERIMETER]", (rx + 5, max(ry - 6, 16)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

            # 2. Draw Trajectory Trails & Top Alert Banner (SECURITY mode)
            if self.mode == "SECURITY":
                for tid, pts in list(self.track_history.items()):
                    if len(pts) > 1:
                        for i in range(1, len(pts)):
                            cv2.line(frame, pts[i - 1], pts[i], (255, 200, 0), 1)

                if dets.get("any_intrusion"):
                    alert_col = (0, 0, 255) if self.flash_state else (0, 50, 180)
                    cv2.rectangle(frame, (20, 15), (420, 55), alert_col, -1)
                    cv2.putText(frame, "RESTRICTED ZONE INTRUSION!", (30, 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # 3. Draw Telemetry HUD & Status Card (ENERGY mode)
            if self.mode == "ENERGY":
                hud_w, hud_h = 360, 130
                h, w = frame.shape[:2]
                x1, y1 = w - hud_w - 15, 15
                x2, y2 = w - 15, y1 + hud_h
                hud_bg = frame.copy()
                cv2.rectangle(hud_bg, (x1, y1), (x2, y2), (20, 20, 20), -1)
                cv2.addWeighted(hud_bg, 0.80, frame, 0.20, 0, frame)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1)

                occ_count = dets.get("occupancy", 0)
                occ_str = "OCCUPIED" if occ_count > 0 else "EMPTY"
                occ_color = (0, 255, 0) if occ_count > 0 else (0, 70, 255)
                cv2.putText(frame, f"Occupancy: {occ_count} [{occ_str}]", (x1 + 10, y1 + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, occ_color, 2)

                lights_on = dets.get("lights_on", False)
                illum = dets.get("lux_level", 0.0)
                light_str = f"Lights: ON ({illum:.0f} Lux)" if lights_on else "Lights: OFF"
                light_col = (0, 215, 255) if lights_on else (180, 180, 180)
                cv2.putText(frame, light_str, (x1 + 10, y1 + 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, light_col, 1)

                fans_active = dets.get("fans_active", False)
                fan_str = "Ceiling Fans: ROTATING" if fans_active else "Ceiling Fans: STOPPED"
                fan_col = (0, 165, 255) if fans_active else (180, 180, 180)
                cv2.putText(frame, fan_str, (x1 + 10, y1 + 68),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.44, fan_col, 1)

                relay_cut = dets.get("relay_cut_off", False)
                relay_txt = "RELAY: CUT-OFF (POWER SAVED)" if relay_cut else "RELAY: CLOSED (ACTIVE)"
                relay_col = (0, 69, 255) if relay_cut else (100, 255, 100)
                cv2.putText(frame, relay_txt, (x1 + 10, y1 + 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, relay_col, 1)

                saved_kwh = dets.get("kwh_saved", 0.0)
                cv2.putText(frame, f"Total Saved: {saved_kwh:.4f} kWh", (x1 + 10, y1 + 112),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

                if occ_count == 0:
                    banner_h = 65
                    by1 = h - banner_h - 20
                    by2 = h - 20
                    bx1 = 15
                    bx2 = w - 15

                    alert_banner = frame.copy()
                    en_status = dets.get("energy_status", "NORMAL")
                    banner_bg = (0, 0, 160) if en_status == "ENERGY_WASTAGE" else (20, 60, 120)
                    cv2.rectangle(alert_banner, (bx1, by1), (bx2, by2), banner_bg, -1)
                    cv2.addWeighted(alert_banner, 0.85, frame, 0.15, 0, frame)
                    border_c = (0, 0, 255) if en_status == "ENERGY_WASTAGE" else (0, 255, 255)
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), border_c, 2)

                    hdr_txt = "SMART SENTINEL: CLASSROOM EMPTY - AUTOMATED RELAY CUT-OFF!" if en_status == "ENERGY_WASTAGE" else "SMART SENTINEL: NO ONE IN CLASSROOM (COUNTDOWN ACTIVE)"
                    cv2.putText(frame, hdr_txt, (bx1 + 12, by1 + 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2)

                    actions = dets.get("active_actions", [])
                    action_detail = " | ".join(actions) if actions else "All Appliances Already OFF"
                    if relay_cut:
                        action_detail += " => [AUTOMATED CUT-OFF EXECUTED]"
                    cv2.putText(frame, f"Action: {action_detail}", (bx1 + 12, by1 + 48),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                
            # 4. Draw Person Bounding Boxes
            for b in dets.get("boxes", []):
                x1, y1, x2, y2 = b["coords"]
                color = b["color"]
                text = b["text"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, text, (x1, max(20, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                
            cv2.putText(frame, f"OCCUPANCY: {dets.get('occupancy', 0)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if self.mode not in ("EXAM", "EXAM_FRONT", "EXAM_REAR"):
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
