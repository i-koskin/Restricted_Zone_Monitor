import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import List, Dict, Tuple
from config import Config


class PersonTracker:
    def __init__(self):
        self.yolo_model = YOLO(Config.YOLO_MODEL)
        self.deepsort_tracker = DeepSort(
            max_age=Config.MAX_AGE,
            n_init=Config.MIN_HITS,
            max_cosine_distance=Config.MAX_COSINE_DISTANCE,
            nn_budget=Config.NN_BUDGET
        )

    def detect_and_track(self, frame: np.ndarray) -> List[Dict]:
        """Detect people using YOLO and track with DeepSORT"""
        results = []

        # YOLO detection
        yolo_results = self.yolo_model(
            frame,
            conf=Config.CONFIDENCE_THRESHOLD,
            iou=Config.IOU_THRESHOLD,
            imgsz=Config.IMG_SIZE,
            half=Config.USE_HALF_PRECISION,
            device='cuda' if Config.USE_CUDA else 'cpu',
            verbose=False
        )[0]

        detections = []

        for box in yolo_results.boxes:
            if box.cls.item() == 0:  # person class
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = box.conf.item()

                detections.append(
                    ([x1, y1, x2 - x1, y2 - y1], confidence, 'person'))

        # DeepSORT tracking
        tracks = self.deepsort_tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_ltrb()
            x1, y1, x2, y2 = map(int, bbox)
            conf_value = track.get_det_conf() if hasattr(track, 'get_det_conf') else None

            results.append({
                'track_id': track_id,
                'bbox': (x1, y1, x2, y2),
                'class_name': 'person',
                'confidence': round(conf_value, 2) if conf_value is not None else None
            })

        return results


class Visualization:
    @staticmethod
    def draw_zones(frame: np.ndarray, zones: List[List[Tuple[int, int]]]):
        """Draw restricted zones on frame"""
        for zone in zones:
            overlay = frame.copy()
            pts = np.array(zone, np.int32)
            cv2.fillPoly(frame, [pts], Config.ZONE_COLOR)
            cv2.addWeighted(frame, 0.6, overlay, 0.4, 0, frame)
            cv2.polylines(frame, [pts], True, Config.ZONE_COLOR, 2)

            # Add zone label
            if len(zone) > 0:
                label_pos = (zone[0][0], zone[0][1] - 10)
                cv2.putText(frame, "Restricted Zone", label_pos,
                            Config.FONT, Config.FONT_SCALE, Config.ZONE_COLOR, Config.TEXT_THICKNESS)

    @staticmethod
    def draw_tracks(frame: np.ndarray, tracks: List[Dict], alert_manager):
        """Draw bounding boxes and IDs on frame with alarm status"""
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            conf = track['confidence']
            x1, y1, x2, y2 = bbox

            # Get alarm status
            is_alerted, time_remaining = alert_manager.get_alert_status(
                track_id)

            # Choose color based on alert status
            if is_alerted:
                color = Config.ALARM_COLOR
                status_text = "ALARM!"

                # Add time remaining if person left the zone
                if time_remaining > 0:
                    status_text = f"ALARM! {time_remaining:.1f}s"
            else:
                color = (255, 0, 0)  # blue
                status_text = f"ID: {track_id} {conf}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          color, Config.BBOX_THICKNESS)

            # Draw label background
            label = f"{status_text}"
            label_size = cv2.getTextSize(
                label, Config.FONT, Config.FONT_SCALE, Config.TEXT_THICKNESS)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                          (x1 + label_size[0], y1), color, -1)

            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5),
                        Config.FONT, Config.FONT_SCALE, Config.TEXT_COLOR, Config.TEXT_THICKNESS)

    @staticmethod
    def draw_alarm_status(frame: np.ndarray, alert_manager):
        """Draw overall alarm status on frame"""
        stats = alert_manager.get_alert_statistics()

        if stats['total_alerts'] > 0:
            cv2.putText(frame, "ALARM!", (10, 30),
                        Config.FONT, 1, Config.ALARM_COLOR, 2)

        else:
            cv2.putText(frame, "Monitoring...", (10, 30),
                        Config.FONT, 0.7, (0, 255, 0), 2)
