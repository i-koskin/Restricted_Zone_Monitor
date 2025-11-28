import cv2
import json
import torch
from typing import List, Tuple


class Config:
    # YOLO model configuration
    YOLO_MODEL = "yolov8l.pt"
    CONFIDENCE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    IMG_SIZE = 640

    # DeepSORT configuration
    MAX_AGE = 30
    MIN_HITS = 5
    MAX_COSINE_DISTANCE = 0.1
    NN_BUDGET = 100

    # Detection optimization
    USE_HALF_PRECISION = True
    USE_CUDA = torch.cuda.is_available()

    # Alert configuration
    ALARM_DURATION = 3  # seconds after leaving zone
    ALARM_COLOR = (0, 0, 255)  # red
    ZONE_COLOR = (0, 255, 0)  # green
    TEXT_COLOR = (255, 255, 255)  # white

    # Visualization
    BBOX_THICKNESS = 2
    TEXT_THICKNESS = 2
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.7

    @staticmethod
    def load_zones() -> List[List[Tuple[int, int]]]:
        try:
            with open('restricted_zones.json', 'r') as f:
                data = json.load(f)
                return [zone for zone in data.get('zones', [])]
        except FileNotFoundError:
            return []

    @staticmethod
    def save_zones(zones: List[List[Tuple[int, int]]]):
        data = {'zones': zones}
        with open('restricted_zones.json', 'w') as f:
            json.dump(data, f, indent=2)
