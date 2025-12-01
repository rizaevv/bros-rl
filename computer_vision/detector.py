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

    def get_observation(self, frame):
        """
        Processes the frame and returns a simplified observation.
        For now, we can return the raw frame or a processed version.
        Ideally, this would return positions of key objects.
        """
        # For a simple visual RL agent, we might just return the frame itself
        # or a resized grayscale version.
        # Let's return the frame for now, the Env will handle resizing/normalization.
        return frame
