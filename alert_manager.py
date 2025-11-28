import time
from typing import Dict, Set, List, Tuple
from dataclasses import dataclass
from config import Config


@dataclass
class Alert:
    track_id: int
    zone_id: int
    entry_time: float
    last_seen_in_zone: float
    is_active: bool = True


class AlertManager:
    def __init__(self):
        self.active_alerts: Dict[int, Alert] = {}
        self.track_positions: Dict[int, Tuple[float, float]] = {}

    def check_point_in_polygon(self, point: Tuple[float, float], polygon: List[Tuple[int, int]]) -> bool:
        """Check if a point is inside a polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / \
                                (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def get_bbox_center(self, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Get center point of bounding box"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return center_x, center_y

    def check_zone_penetration(self, track_id: int, bbox: Tuple[float, float, float, float],
                               zones: List[List[Tuple[int, int]]]) -> Tuple[bool, int]:
        """Check if track center is inside any restricted zone"""
        center_point = self.get_bbox_center(bbox)
        self.track_positions[track_id] = center_point

        for zone_id, zone in enumerate(zones):
            if self.check_point_in_polygon(center_point, zone):
                return True, zone_id

        return False, -1

    def update_alerts(self, tracks: List[Dict], zones: List[List[Tuple[int, int]]]) -> Set[int]:
        """Update alerts based on current tracks and zones"""
        current_time = time.time()
        active_track_ids = set(track['track_id'] for track in tracks)
        current_zone_occupancy = {}

        # Check for new zone penetrations
        for track in tracks:
            track_id = track['track_id']
            bbox = track['bbox']

            # Check if center point is in any zone
            penetrated, zone_id = self.check_zone_penetration(
                track_id, bbox, zones)

            if penetrated:
                if zone_id not in current_zone_occupancy:
                    current_zone_occupancy[zone_id] = set()
                current_zone_occupancy[zone_id].add(track_id)

                # Create new alert or update existing one
                if track_id not in self.active_alerts:
                    self.active_alerts[track_id] = Alert(
                        track_id=track_id,
                        zone_id=zone_id,
                        entry_time=current_time,
                        last_seen_in_zone=current_time,
                        is_active=True
                    )
                else:
                    # Update existing alert - still in zone
                    self.active_alerts[track_id].last_seen_in_zone = current_time
                    self.active_alerts[track_id].is_active = True

        # Handle tracks that left zones or disappeared
        alerts_to_remove = []
        for track_id, alert in self.active_alerts.items():
            if track_id not in active_track_ids:
                # Track disappeared - check if alarm duration expired
                time_since_last_seen = current_time - alert.last_seen_in_zone
                if time_since_last_seen >= Config.ALARM_DURATION:
                    alerts_to_remove.append(track_id)
            else:
                # Track is still active
                current_track = next(
                    (t for t in tracks if t['track_id'] == track_id), None)
                if current_track:
                    penetrated, _ = self.check_zone_penetration(
                        track_id, current_track['bbox'], zones)
                    if not penetrated:
                        # Track left the zone - update last seen time but keep alarm active
                        alert.is_active = False

                        # Check if alarm duration expired since leaving zone
                        if track_id in self.track_positions:
                            # Use current time for time calculation
                            time_since_left = current_time - alert.last_seen_in_zone
                            if time_since_left >= Config.ALARM_DURATION:
                                alerts_to_remove.append(track_id)

        # Remove expired alerts
        for track_id in alerts_to_remove:
            if track_id in self.active_alerts:
                del self.active_alerts[track_id]
            if track_id in self.track_positions:
                del self.track_positions[track_id]

        # Clean up positions for disappeared tracks
        disappeared_tracks = set(
            self.track_positions.keys()) - active_track_ids
        for track_id in disappeared_tracks:
            if track_id in self.track_positions:
                del self.track_positions[track_id]

        return set(self.active_alerts.keys())

    def get_alerted_tracks(self) -> Set[int]:
        """Get all tracks with active alarms"""
        return set(self.active_alerts.keys())

    def get_alert_status(self, track_id: int) -> Tuple[bool, float]:
        """Get alarm status and time remaining for a track"""
        if track_id not in self.active_alerts:
            return False, 0

        alert = self.active_alerts[track_id]
        current_time = time.time()

        if alert.is_active:
            return True, 0  # Still in zone, alarm continues indefinitely

        # Calculate time remaining for alarm after leaving zone
        time_since_left = current_time - alert.last_seen_in_zone
        time_remaining = max(0, Config.ALARM_DURATION - time_since_left)

        return time_remaining > 0, time_remaining

    def get_alert_statistics(self) -> Dict:
        """Get statistics about current alerts"""
        # current_time = time.time()
        # active_in_zone = 0
        # active_after_leave = 0

        # for alert in self.active_alerts.values():
        #     if alert.is_active:
        #         active_in_zone += 1
        #     else:
        #         time_since_left = current_time - alert.last_seen_in_zone
        #         if time_since_left < Config.ALARM_DURATION:
        #             active_after_leave += 1

        return {
            'total_alerts': len(self.active_alerts),
            # 'active_in_zone': active_in_zone,
            # 'active_after_leave': active_after_leave,
            # 'track_positions': len(self.track_positions)
        }
