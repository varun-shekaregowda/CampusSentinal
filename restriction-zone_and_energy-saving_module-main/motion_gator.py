"""
motion_gator.py
---------------
Edge CPU optimization layer using MOG2 Background Subtraction
(derived from Python-Movement-Tracker-master).
Gating ignores static frames to save CPU / battery consumption,
while a periodic heartbeat ensures stationary occupants are never missed.
"""
import cv2
import numpy as np
import config

class MotionGator:
    def __init__(self, min_area=config.MOTION_MIN_AREA, heartbeat=config.MOTION_HEARTBEAT_FRAMES):
        self.min_area = min_area
        self.heartbeat = heartbeat
        self.frame_counter = 0
        self.back_sub = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True,
            varThreshold=50,
            history=500
        )
        self.kernel = np.ones((5, 5), np.uint8)

    def check_motion(self, frame) -> tuple[bool, np.ndarray, list]:
        """
        Processes frame for motion gating.
        Returns:
            should_run_yolo (bool): True if motion exceeds threshold or heartbeat reached
            fg_mask (np.ndarray): Binary foreground mask
            motion_boxes (list): Bounding rectangles [(x, y, w, h)] for visualization
        """
        self.frame_counter += 1
        is_heartbeat = (self.frame_counter % self.heartbeat == 0)

        # Apply background subtraction
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
