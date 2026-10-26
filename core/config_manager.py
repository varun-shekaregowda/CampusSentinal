import os
import json
import uuid

class ConfigManager:
    def __init__(self):
        self.config_file = os.path.join("storage", "config.json")
        self.cameras = []
        self.load_config()

    def load_config(self):
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.cameras = json.load(f)
                return
            except:
                pass
                
        # Default config if file doesn't exist or is corrupted
        self.cameras = [
            {
                "id": str(uuid.uuid4())[:8],
                "name": "Exam Hall Front",
                "url": "0",  # Default local webcam, stored as string
                "mode": "EXAM_FRONT",
                "roi": [] 
            },
            {
                "id": str(uuid.uuid4())[:8],
                "name": "Exam Hall Rear",
                "url": "storage/model school-Channel 51.MP4",
                "mode": "EXAM_REAR",
                "roi": [] 
            },
            {
                "id": str(uuid.uuid4())[:8],
                "name": "Terrace Restricted Zone",
                "url": "restriction-zone_and_energy-saving_module-main/terrace.mp4", 
                "mode": "SECURITY",
                "roi": [(160, 100), (440, 100), (440, 420), (160, 420)] 
            },
            {
                "id": str(uuid.uuid4())[:8],
                "name": "Smart Classroom 101",
                "url": "restriction-zone_and_energy-saving_module-main/classroom.mp4", 
                "mode": "ENERGY",
                "roi": [] 
            }
        ]
        self.save_config()

    def save_config(self):
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.cameras, f, indent=4)

    def get_cameras(self):
        return self.cameras
    
    def get_camera_config(self, cam_id):
        for cam in self.cameras:
            if cam['id'] == cam_id:
                return cam
        return None

    def add_camera(self, name, url, mode, roi=None):
        if roi is None:
            roi = []
        new_cam = {
            "id": str(uuid.uuid4())[:8],
            "name": name,
            "url": url,
            "mode": mode,
            "roi": roi
        }
        self.cameras.append(new_cam)
        self.save_config()
        return new_cam

    def update_camera(self, cam_id, updates):
        for i, cam in enumerate(self.cameras):
            if cam['id'] == cam_id:
                self.cameras[i].update(updates)
                self.save_config()
                return self.cameras[i]
        return None

    def delete_camera(self, cam_id):
        self.cameras = [c for c in self.cameras if c['id'] != cam_id]
        self.save_config()

config_manager = ConfigManager()
