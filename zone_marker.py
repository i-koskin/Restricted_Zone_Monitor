import cv2
import numpy as np
from typing import List, Tuple
from config import Config


class ZoneMarker:
    def __init__(self):
        self.current_zone_points: List[Tuple[int, int]] = []
        self.zones: List[List[Tuple[int, int]]] = []
        self.current_frame = None
        self.window_name = "Mark Restricted Zones: Left click: add points, Right click: finish zone, 'q' or Esc: quit, 'c': clear current point, 'd': delete last zone"

    def mark_zones_callback(self, event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_zone_points.append((x, y))

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_zone_points) >= 3:
                self.zones.append(self.current_zone_points.copy())
                self.current_zone_points = []
                Config.save_zones(self.zones)
                print(f"Zone saved. Total zones: {len(self.zones)}")
            else:
                print("Need at least 3 points to create a zone")

    def draw_current_zone(self):
        temp_frame = self.current_frame.copy()

        # Draw existing zones
        for zone in self.zones:
            overlay = temp_frame.copy()
            pts = np.array(zone, np.int32)
            cv2.fillPoly(temp_frame, [pts], Config.ZONE_COLOR)
            cv2.addWeighted(temp_frame, 0.6, overlay, 0.4, 0, temp_frame)
            cv2.polylines(temp_frame, [pts],
                          True, Config.ZONE_COLOR, 2)

         # Draw current zone being marked
        if len(self.current_zone_points) > 0:
            for i, point in enumerate(self.current_zone_points):
                cv2.circle(temp_frame, point, 5, Config.ZONE_COLOR, -1)
                cv2.putText(temp_frame, str(i+1),
                            (point[0]+10, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            if len(self.current_zone_points) > 1:
                for i in range(len(self.current_zone_points) - 1):
                    cv2.line(temp_frame, self.current_zone_points[i],
                             self.current_zone_points[i+1], Config.ZONE_COLOR, 2)

            if len(self.current_zone_points) >= 3:
                cv2.polylines(temp_frame, [np.array(
                    self.current_zone_points)], True, Config.ZONE_COLOR, 2)

        cv2.imshow(self.window_name, temp_frame)

    def setup_zones(self, frame_path: str) -> List[List[Tuple[int, int]]]:
        """Interactive polygon zone marking setup"""

        self.current_frame = cv2.imread(
            frame_path) if frame_path.endswith(('.jpg', '.png')) else None
        if self.current_frame is None:
            cap = cv2.VideoCapture(frame_path)
            ret, self.current_frame = cap.read()
            cap.release()
            if not ret:
                raise ValueError("Could not read frame from source")

        self.zones = Config.load_zones()
        self.current_zone_points = []

        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mark_zones_callback)

        while True:
            self.draw_current_zone()

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # Quit from zone marking mode: 'q' or Esc
                break

            elif key == ord('c'):  # Clear current points
                if self.current_zone_points:
                    self.current_zone_points.pop()

            elif key == ord('d'):  # Delete last zone
                if self.zones:
                    self.zones.pop()
                    Config.save_zones(self.zones)
                    print(f"Last zone deleted. Total zones: {len(self.zones)}")

        cv2.destroyAllWindows()
        return self.zones
