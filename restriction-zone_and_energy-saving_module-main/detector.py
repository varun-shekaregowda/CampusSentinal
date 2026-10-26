"""
detector.py
------------
Integrated Edge AI Engine for:
- Module 2: Restricted Campus Zone Sentinel
    * Multi-ROI polygonal perimeters (e.g., terrace access, high-voltage yards)
    * Directional centroid tracking & boundary entry vectors
    * False alarm rejection (bounding-box size, aspect ratio, jitter filtering)
    * Loitering escalation
- Module 3: Smart Hall & Auditorium Energy Wastage Sentinel
    * Human occupancy counter
    * Active illumination detection
    * Zero-occupancy timer logic & state transitions
    * Automated relay cut-off simulation & kWh savings computation
- Motion Gating integration for edge CPU efficiency
"""
import time
import json
import math
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from ultralytics import YOLO

import config
from database import log_security_event_db, log_energy_event_db
from motion_gator import MotionGator


@dataclass
class TrackState:
    """Tracks state and trajectory for a single person."""
    current_point: Tuple[int, int] = (0, 0)
    history: List[Tuple[int, int]] = field(default_factory=list)
    inside_zones: Dict[str, bool] = field(default_factory=dict)
    entry_times: Dict[str, float] = field(default_factory=dict)
    last_alert_time: float = 0.0
    loiter_flagged: Dict[str, bool] = field(default_factory=dict)
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    aspect_ratio: float = 0.0
    area: float = 0.0


class CentroidTracker:
    """
    Nearest-neighbour centroid & trajectory tracker.
    Stores historical trajectory points to compute velocity and directional vectors.
    """
    def __init__(self, max_distance=80, max_missed_frames=20, max_history=15):
        self.next_id = 0
        self.objects: Dict[int, Tuple[Tuple[int, int], int, List[Tuple[int, int]]]] = {}
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self.max_history = max_history

    def update(self, points: List[Tuple[int, int]]) -> List[int]:
        assigned = [None] * len(points)
        unmatched = set(range(len(points)))

        for oid, (centroid, missed, history) in list(self.objects.items()):
            best_idx, best_dist = None, self.max_distance
            for idx in unmatched:
                dx = points[idx][0] - centroid[0]
                dy = points[idx][1] - centroid[1]
                dist = math.hypot(dx, dy)
                if dist < best_dist:
                    best_dist, best_idx = dist, idx

            if best_idx is not None:
                new_pt = points[best_idx]
                history.append(new_pt)
                if len(history) > self.max_history:
                    history.pop(0)
                self.objects[oid] = (new_pt, 0, history)
                assigned[best_idx] = oid
                unmatched.discard(best_idx)
            else:
                self.objects[oid] = (centroid, missed + 1, history)

        # Drop tracks missed for too long
        for oid in [k for k, v in self.objects.items() if v[1] > self.max_missed_frames]:
            del self.objects[oid]

        # Assign new IDs for unmatched detections
        for idx in unmatched:
            oid = self.next_id
            self.next_id += 1
            pt = points[idx]
            self.objects[oid] = (pt, 0, [pt])
            assigned[idx] = oid

        return assigned

    def get_trajectory(self, track_id: int) -> List[Tuple[int, int]]:
        if track_id in self.objects:
            return self.objects[track_id][2]
        return []


class CampusEdgeSentinel:
    """
    Unified Edge Processing Pipeline:
    Combines Motion Gating, YOLOv8-Nano, Module 2 (Restricted Zones),
    and Module 3 (Energy Wastage Sentinel).
    """

    def __init__(self, model_path=config.MODEL_PATH, zone_path=config.ZONE_FILE, enable_gating=config.ENABLE_MOTION_GATING):
        self.model = YOLO(model_path)
        self.zones = self._load_zones(zone_path)
        self.tracker = CentroidTracker()
        self.track_states: Dict[int, TrackState] = {}
        self.enable_gating = enable_gating
        self.motion_gator = MotionGator() if enable_gating else None

        # Module 3: Energy Sentinel State
        self.occupancy_count = 0
        self.empty_start_time: Optional[float] = None
        self.energy_status = "NORMAL"  # "NORMAL", "COUNTDOWN", "ENERGY_WASTAGE"
        self.relay_cut_off = False
        self.lights_relay_cutoff = False
        self.fans_relay_cutoff = False
        self.lights_on = False
        self.fans_rotating = False
        self.current_illumination = 0.0
        self.active_actions: List[str] = []
        self.prev_gray_frame: Optional[np.ndarray] = None
        self.relay_triggered_time: Optional[float] = None
        self.total_energy_saved_kwh = 0.0
        self.last_energy_log_time = 0.0
        self.flash_state = False
        self.last_flash_time = time.time()

        # Cached detections when gating skips YOLO
        self.last_boxes_xyxy = []
        self.last_track_ids = []

    def _load_zones(self, path: str) -> List[Dict]:
        """Loads single-zone or multi-ROI zone configurations."""
        zones = []
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "zones" in data:
                    for z in data["zones"]:
                        poly = np.array(z["points"], dtype=np.int32)
                        zones.append({
                            "id": z.get("id", f"Zone-{len(zones)+1}"),
                            "name": z.get("name", "Restricted Perimeter"),
                            "polygon": poly,
                            "color": tuple(z.get("color", [0, 0, 255]))
                        })
                elif "zone" in data:
                    poly = np.array(data["zone"], dtype=np.int32)
                    zones.append({
                        "id": "Zone-1",
                        "name": "High Voltage / Perimeter Yard",
                        "polygon": poly,
                        "color": (0, 0, 255)
                    })
            except Exception as e:
                print(f"[Detector] Warning reading zone file {path}: {e}")

        if not zones:
            # Fallback default perimeter zone
            default_poly = np.array([[180, 100], [480, 100], [480, 420], [180, 420]], dtype=np.int32)
            zones.append({
                "id": "Zone-1",
                "name": "Restricted Terrace / Perimeter Yard",
                "polygon": default_poly,
                "color": (0, 0, 255)
            })
        return zones

    def set_custom_zone(self, points: List[List[int]], name="Custom Restricted Zone", color=(0, 0, 255)):
        """Allows dynamically applying a user-defined zone polygon at runtime."""
        poly = np.array(points, dtype=np.int32)
        self.zones = [{
            "id": "Zone-Custom",
            "name": name,
            "polygon": poly,
            "color": color
        }]
        print(f"[Detector] Custom zone updated: '{name}' with {len(points)} vertices.")

    def _filter_false_alarm(self, box: Tuple[int, int, int, int]) -> Tuple[bool, str]:
        """
        Filters out natural false alarms (wind-blown leaves, tree shadows, pets)
        using bounding-box area, speed, and human aspect ratio.
        """
        x1, y1, x2, y2 = box
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        area = w * h
        aspect_ratio = h / float(w)

        if area < config.MIN_PERSON_BBOX_AREA:
            return False, f"Area too small ({area:.0f}px - leaf/debris)"
        if area > config.MAX_PERSON_BBOX_AREA:
            return False, f"Area too large ({area:.0f}px - wide shadow)"
        if aspect_ratio < config.MIN_HUMAN_ASPECT_RATIO:
            return False, f"Flat aspect ratio ({aspect_ratio:.2f} - not standing human)"
        if aspect_ratio > config.MAX_HUMAN_ASPECT_RATIO:
            return False, f"Distorted aspect ratio ({aspect_ratio:.2f})"

        return True, "Valid Human Detection"

    def _compute_directional_entry(self, trajectory: List[Tuple[int, int]], zone_poly: np.ndarray) -> Tuple[bool, Tuple[int, int]]:
        """
        Checks if movement vector crosses from outside into the zone.
        Returns (is_directional_entry, vector_displacement).
        """
        if len(trajectory) < 2:
            return False, (0, 0)

        # Look back 3-5 frames for stable movement vector
        start_pt = trajectory[max(0, len(trajectory) - 5)]
        end_pt = trajectory[-1]
        dx = end_pt[0] - start_pt[0]
        dy = end_pt[1] - start_pt[1]
        disp = math.hypot(dx, dy)

        was_outside = cv2.pointPolygonTest(zone_poly, (int(start_pt[0]), int(start_pt[1])), False) < 0
        now_inside = cv2.pointPolygonTest(zone_poly, (int(end_pt[0]), int(end_pt[1])), False) >= 0

        is_directional = was_outside and now_inside and (disp >= config.MIN_DIRECTIONAL_DISPLACEMENT)
        return is_directional, (int(dx), int(dy))

    def _calculate_illumination(self, frame: np.ndarray) -> float:
        """
        Calculates average frame luminance to verify if lighting is ON.
        Uses perceptual grayscale conversion.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.current_illumination = float(np.mean(gray))
        return self.current_illumination

    def _detect_fan_rotation(self, frame: np.ndarray) -> bool:
        """
        Detects ceiling fan rotation by checking motion difference in the top ceiling ROI (top 35% of frame).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        ceiling_h = int(h * 0.35)  # Ceiling region where fans are mounted

        is_rotating = False
        if self.prev_gray_frame is not None and self.prev_gray_frame.shape == gray.shape:
            diff = cv2.absdiff(self.prev_gray_frame[:ceiling_h, :], gray[:ceiling_h, :])
            # Percentage of ceiling pixels undergoing active motion
            motion_ratio = float(np.mean(diff > 18))
            is_rotating = (motion_ratio > 0.005)

        self.prev_gray_frame = gray
        self.fans_rotating = is_rotating
        return is_rotating

    def update_energy_sentinel(self, frame: np.ndarray, occupancy: int, now: float) -> List[Dict]:
        """
        Module 3 Logic:
        Monitors room occupancy, lighting status, and ceiling fan rotation.
        If occupancy == 0 for >= timeout:
        Identifies exact devices wasting power:
          - "Turn OFF Lights (Lights are ON)"
          - "Turn OFF Ceiling Fans (Fans are rotating)"
        Triggers automated relay cut-offs and logs power savings.
        """
        events = []
        illumination = self._calculate_illumination(frame)
        self.lights_on = (illumination >= config.ACTIVE_ILLUMINATION_THRESHOLD)
        self.fans_rotating = self._detect_fan_rotation(frame)

        if occupancy > 0:
            # Room is occupied
            self.occupancy_count = occupancy
            self.empty_start_time = None
            self.energy_status = "NORMAL"
            self.relay_cut_off = False
            self.lights_relay_cutoff = False
            self.fans_relay_cutoff = False
            self.active_actions = []
            self.appliance_actions = []
        else:
            # Room is empty (0 occupants)
            self.occupancy_count = 0

            # Determine specific devices wasting energy
            appliance_actions = []
            active_watts = 0.0
            if self.lights_on:
                appliance_actions.append("Turn OFF Lights (Lights are ON)")
                if not self.lights_relay_cutoff:
                    active_watts += getattr(config, "LIGHT_WATTS", 1200.0)
            if self.fans_rotating:
                appliance_actions.append("Turn OFF Ceiling Fans (Fans are rotating)")
                if not self.fans_relay_cutoff:
                    active_watts += getattr(config, "FAN_WATTS", 300.0)

            self.appliance_actions = appliance_actions
            self.active_actions = appliance_actions

            if appliance_actions and not self.relay_cut_off:
                if self.empty_start_time is None:
                    self.empty_start_time = now

                empty_duration = now - self.empty_start_time

                if empty_duration >= config.ZERO_OCCUPANCY_TIMEOUT_SECONDS:
                    # Timeout exceeded -> ENERGY WASTAGE DETECTED
                    self.energy_status = "ENERGY_WASTAGE"
                    self.relay_cut_off = True
                    self.lights_relay_cutoff = self.lights_on
                    self.fans_relay_cutoff = self.fans_rotating

                    kwh_saved = (active_watts * (empty_duration / 3600.0)) / 1000.0
                    self.total_energy_saved_kwh += kwh_saved

                    action_str = " & ".join(appliance_actions)
                    detail_msg = f"No one in classroom! Action: {action_str}"

                    event = {
                        "type": "RELAY_CUTOFF",
                        "room": config.ROOM_NAME,
                        "occupancy": 0,
                        "illumination": round(illumination, 1),
                        "empty_duration": round(empty_duration, 1),
                        "kwh_saved": round(kwh_saved, 4),
                        "relay": True,
                        "lights_on": self.lights_on,
                        "fans_rotating": self.fans_rotating,
                        "actions": appliance_actions,
                        "detail": detail_msg
                    }
                    events.append(event)
                    log_energy_event_db(
                        room_name=config.ROOM_NAME,
                        status="ENERGY_WASTAGE_RELAY_CUTOFF",
                        occupancy=0,
                        illumination=round(illumination, 1),
                        empty_duration=round(empty_duration, 1),
                        kwh_saved=round(kwh_saved, 4),
                        relay_triggered=True,
                        detail=detail_msg
                    )
                else:
                    self.energy_status = "COUNTDOWN"
            elif not appliance_actions:
                self.energy_status = "ALL_DEVICES_OFF"
                self.empty_start_time = None

        return events

    def process_frame(self, frame: np.ndarray, enable_module2=True, enable_module3=True) -> Tuple[np.ndarray, List[Dict]]:
        """
        Executes pipeline for a single frame.
        Returns (annotated_frame, list_of_events).
        """
        now = time.time()
        events = []
        should_run_yolo = True

        # Toggle flash state for flashing alerts
        if now - self.last_flash_time > 0.4:
            self.flash_state = not self.flash_state
            self.last_flash_time = now

        # 1. Motion Gating Check
        if self.enable_gating and self.motion_gator is not None:
            has_motion, fg_mask, motion_boxes = self.motion_gator.check_motion(frame)
            should_run_yolo = has_motion

        boxes_xyxy = []
        track_ids = []

        if should_run_yolo:
            # 2. YOLOv8 Person Detection
            results = self.model(
                frame,
                classes=[0],  # class 0 = person in COCO
                conf=config.CONFIDENCE_THRESHOLD,
                imgsz=640,
                verbose=False
            )
            raw_boxes = [tuple(map(int, b.xyxy[0])) for b in results[0].boxes]

            # 3. False Alarm Filtering
            valid_boxes = []
            valid_foot_points = []
            for b in raw_boxes:
                is_valid, reason = self._filter_false_alarm(b)
                if is_valid:
                    valid_boxes.append(b)
                    foot_pt = (int((b[0] + b[2]) / 2), b[3])
                    valid_foot_points.append(foot_pt)

            boxes_xyxy = valid_boxes
            track_ids = self.tracker.update(valid_foot_points)
            self.last_boxes_xyxy = boxes_xyxy
            self.last_track_ids = track_ids
        else:
            # Reuse previous detections during static frame gating
            boxes_xyxy = self.last_boxes_xyxy
            track_ids = self.last_track_ids

        # Draw Zones (Module 2)
        if enable_module2:
            overlay = frame.copy()
            for z in self.zones:
                cv2.fillPoly(overlay, [z["polygon"]], z["color"])
                cv2.polylines(frame, [z["polygon"]], isClosed=True, color=z["color"], thickness=3)
                # Zone Label Tag with solid dark backdrop for crisp visibility
                tag_x, tag_y = z["polygon"][0]
                label_text = f"[SEC-ROI] {z['name']}"
                cv2.rectangle(frame, (tag_x, max(0, tag_y - 22)), (tag_x + 230, tag_y), (10, 10, 10), -1)
                cv2.putText(frame, label_text, (tag_x + 5, max(tag_y - 6, 16)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            cv2.addWeighted(overlay, 0.22, frame, 0.78, 0, frame)

        any_intrusion = False

        # Evaluate each tracked person
        for (x1, y1, x2, y2), track_id in zip(boxes_xyxy, track_ids):
            foot_point = (int((x1 + x2) / 2), y2)
            state = self.track_states.setdefault(track_id, TrackState())
            state.current_point = foot_point
            state.bbox = (x1, y1, x2, y2)
            trajectory = self.tracker.get_trajectory(track_id)

            person_in_any_zone = False

            if enable_module2:
                for z in self.zones:
                    zid = z["id"]
                    zname = z["name"]
                    inside = cv2.pointPolygonTest(z["polygon"], foot_point, False) >= 0

                    if inside:
                        person_in_any_zone = True
                        any_intrusion = True

                        if not state.inside_zones.get(zid, False):
                            # Directional boundary entry check
                            is_dir, vec = self._compute_directional_entry(trajectory, z["polygon"])
                            state.inside_zones[zid] = True
                            state.entry_times[zid] = now
                            state.loiter_flagged[zid] = False

                            if now - state.last_alert_time > config.ALERT_COOLDOWN_SECONDS:
                                event_type = "DIRECTIONAL_ENTRY" if is_dir else "ZONE_ENTRY"
                                detail_msg = f"Vector ({vec[0]},{vec[1]}) into {zname}" if is_dir else f"Entered {zname}"
                                events.append({
                                    "type": event_type,
                                    "track_id": track_id,
                                    "zone_id": zid,
                                    "zone_name": zname,
                                    "detail": detail_msg,
                                    "bbox": (x1, y1, x2, y2)
                                })
                                state.last_alert_time = now
                        else:
                            # Loitering check
                            entry_t = state.entry_times.get(zid, now)
                            loiter_duration = now - entry_t
                            if loiter_duration > config.LOITER_THRESHOLD_SECONDS and not state.loiter_flagged.get(zid, False):
                                events.append({
                                    "type": "LOITERING",
                                    "track_id": track_id,
                                    "zone_id": zid,
                                    "zone_name": zname,
                                    "duration": round(loiter_duration, 1),
                                    "detail": f"Loitering in {zname} for {loiter_duration:.1f}s",
                                    "bbox": (x1, y1, x2, y2)
                                })
                                state.loiter_flagged[zid] = True
                    else:
                        state.inside_zones[zid] = False

            # Draw person bounding box & ID
            box_color = (0, 0, 255) if person_in_any_zone else (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.circle(frame, foot_point, 4, box_color, -1)

            # Draw trajectory trail
            if len(trajectory) > 1:
                for i in range(1, len(trajectory)):
                    cv2.line(frame, trajectory[i - 1], trajectory[i], (255, 200, 0), 1)

            cv2.putText(frame, f"ID {track_id} [HUMAN]", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)

        # 4. Evaluate Module 3: Energy Wastage Sentinel
        if enable_module3:
            energy_events = self.update_energy_sentinel(frame, len(boxes_xyxy), now)
            events.extend(energy_events)

        # 5. Draw HUD & Status Banners
        self._draw_hud(frame, any_intrusion, enable_module2, enable_module3)

        return frame, events

    def _draw_hud(self, frame: np.ndarray, any_intrusion: bool, enable_mod2: bool, enable_mod3: bool):
        h, w = frame.shape[:2]

        # Top Banner: Security Alert (Module 2)
        if enable_mod2 and any_intrusion:
            alert_color = (0, 0, 255) if self.flash_state else (0, 50, 180)
            cv2.rectangle(frame, (20, 15), (380, 55), alert_color, -1)
            cv2.putText(frame, "RESTRICTED ZONE INTRUSION!", (30, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        # Top Right HUD: Module 3 Energy Mode
        if enable_mod3:
            hud_w, hud_h = 350, 130
            x1, y1 = w - hud_w - 15, 15
            x2, y2 = w - 15, y1 + hud_h

            # HUD Background Card
            hud_bg = frame.copy()
            cv2.rectangle(hud_bg, (x1, y1), (x2, y2), (20, 20, 20), -1)
            cv2.addWeighted(hud_bg, 0.80, frame, 0.20, 0, frame)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1)

            # Room Occupancy
            occ_str = "OCCUPIED" if self.occupancy_count > 0 else "EMPTY"
            occ_color = (0, 255, 0) if self.occupancy_count > 0 else (0, 70, 255)
            cv2.putText(frame, f"Occupancy: {self.occupancy_count} [{occ_str}]", (x1 + 10, y1 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, occ_color, 2)

            # Lighting State
            light_str = f"Lights: ON ({self.current_illumination:.0f} Lux)" if self.lights_on else "Lights: OFF"
            light_col = (0, 215, 255) if self.lights_on else (180, 180, 180)
            cv2.putText(frame, light_str, (x1 + 10, y1 + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, light_col, 1)

            # Fan Rotation State
            fan_str = "Ceiling Fans: ROTATING" if self.fans_rotating else "Ceiling Fans: STOPPED"
            fan_col = (0, 165, 255) if self.fans_rotating else (180, 180, 180)
            cv2.putText(frame, fan_str, (x1 + 10, y1 + 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, fan_col, 1)

            # Relay State & Energy Saved
            relay_txt = "RELAY: CUT-OFF (POWER SAVED)" if self.relay_cut_off else "RELAY: CLOSED (ACTIVE)"
            relay_col = (0, 69, 255) if self.relay_cut_off else (100, 255, 100)
            cv2.putText(frame, relay_txt, (x1 + 10, y1 + 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, relay_col, 1)

            cv2.putText(frame, f"Total Saved: {self.total_energy_saved_kwh:.4f} kWh", (x1 + 10, y1 + 112),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

            # Prominent Action Banner across frame when Classroom is Empty:
            if self.occupancy_count == 0:
                banner_h = 65
                by1 = h - banner_h - 20
                by2 = h - 20
                bx1 = 15
                bx2 = w - 15

                alert_banner = frame.copy()
                banner_bg_color = (0, 0, 160) if self.flash_state else (20, 20, 80)
                cv2.rectangle(alert_banner, (bx1, by1), (bx2, by2), banner_bg_color, -1)
                cv2.addWeighted(alert_banner, 0.85, frame, 0.15, 0, frame)
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)

                cv2.putText(frame, "SMART SENTINEL: NO ONE IN CLASSROOM!", (bx1 + 12, by1 + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.54, (0, 255, 255), 2)

                action_items = []
                if self.lights_on:
                    action_items.append("Turn OFF Lights (Lights are ON)")
                if self.fans_rotating:
                    action_items.append("Turn OFF Ceiling Fans (Fans are rotating)")
                if not action_items:
                    action_items.append("All Appliances Already OFF")

                action_detail = " | ".join(action_items)
                if self.relay_cut_off:
                    action_detail += " => [AUTOMATED CUT-OFF TRIGGERED]"

                cv2.putText(frame, f"Action: {action_detail}", (bx1 + 12, by1 + 48),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        # Bottom Left: Pipeline Status (Motion Gating)
        gating_status = "ENABLED (IDLE SKIP)" if self.enable_gating else "DISABLED"
        cv2.putText(frame, f"Edge Gating: {gating_status} | Model: YOLOv8n CPU", (15, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)


# Compatibility alias so existing scripts (main.py, app.py) don't break
ZoneMonitor = CampusEdgeSentinel
