import mss
import numpy as np
import cv2
from ultralytics import YOLO
import sys
import os

# Add project root to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class ScreenCapture:
    def __init__(self, region=None):
        self.sct = mss.mss()
        self.region = region if region else config.SCREEN_REGION

    def capture(self):
        """Captures the screen and returns a numpy array (BGR)."""
        screenshot = self.sct.grab(self.region)
        frame = np.array(screenshot)
        # Remove alpha channel if present
        frame = frame[:, :, :3]
        return frame

class Detector:
    def __init__(self, model_path=None):
        self.model_path = model_path if model_path else config.YOLO_MODEL_PATH
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            print("Using default 'yolov8n.pt' as fallback if available or ensure the model path is correct.")
            self.model = YOLO("yolov8n.pt") # Fallback

    def detect(self, frame):
        """
        Runs object detection on the frame.
        Returns the results object from YOLO.
        """
        results = self.model(frame, verbose=False)
        return results

    def get_detections(self, frame):
        """
        Returns a list of detected objects with their class, confidence, and bounding box.
        Format: [{'class': 'ball', 'conf': 0.9, 'bbox': [x1, y1, x2, y2], 'center': (cx, cy)}, ...]
        """
        results = self.detect(frame)
        detections = []
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = config.YOLO_CLASSES.get(cls_id, str(cls_id))
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                detections.append({
                    'class': cls_name,
                    'class_id': cls_id,
                    'conf': conf,
                    'bbox': xyxy,
                    'center': (center_x, center_y)
                })
        return detections
