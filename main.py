import numpy as np
import cv2
import argparse
from zone_marker import ZoneMarker
from tracker import PersonTracker, Visualization
from alert_manager import AlertManager
from config import Config
from typing import Set
from datetime import datetime


class RestrictedZoneMonitor:
    def __init__(self):
        self.tracker = PersonTracker()
        self.alert_manager = AlertManager()
        self.zones = []

    def load_zones(self):
        """Load restricted zones from configuration"""
        self.zones = Config.load_zones()
        if not self.zones:
            print("No restricted zones found. Please mark zones first.")
            return False
        return True

    def mark_zones_interactively(self, video_source: str):
        """Interactive zone marking interface"""
        zone_marker = ZoneMarker()
        self.zones = zone_marker.setup_zones(video_source)
        print(f"Marked restricted zones")

    def draw_additional_alerts(self, frame: np.ndarray, alerted_tracks: Set[int]):
        """Draw additional alert information"""
        if alerted_tracks:
            # Draw a red border around the frame when there are active alerts
            cv2.rectangle(
                frame, (0, 0), (frame.shape[1], frame.shape[0]), Config.ALARM_COLOR, 10)

    def process_video(self, video_source: str, output_path: str = None):
        """Process video stream for intrusion detection"""
        if not self.load_zones():
            return

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {video_source}")
            return

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Setup video writer if output path is provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        else:
            out = None

        print("Starting restricted zone monitoring...")
        print("Press 'q' or Esc to quit, 'p' to pause, 's' to save image")

        paused = False
        frame_count = 0

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                # Track people in frame
                tracks = self.tracker.detect_and_track(frame)

                # Check for zone violations and get alerted tracks
                alerted_tracks = self.alert_manager.update_alerts(
                    tracks, self.zones)

                # Visualize results
                Visualization.draw_zones(frame, self.zones)
                Visualization.draw_tracks(frame, tracks, self.alert_manager)
                Visualization.draw_alarm_status(frame, self.alert_manager)

                # Additional visualization using alerted_tracks directly
                self.draw_additional_alerts(frame, alerted_tracks)

                # Write frame to output
                if out:
                    out.write(frame)

                cv2.imshow(
                    "Restricted Zone Monitoring. Press 'q' or Esc to quit, 'p' to pause, 's' to save image", frame)

            key = cv2.waitKey(1 if not paused else 0) & 0xFF

            if key == ord('q') or key == 27:
                break
            elif key == ord('p'):
                paused = not paused
                status = "Paused" if paused else "Resumed"
                print(status)
                # Show current alert status when pausing
                if paused and alerted_tracks:
                    print(f"Current alerts: {list(alerted_tracks)}")
            elif key == ord('s'):
                # Save current frame with alerts
                filename = f"alert_frame_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Frame saved as {filename}")

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        # Print final statistics
        print("Monitoring stopped.")


def main():
    parser = argparse.ArgumentParser(
        description='Restricted Zone Intrusion Detection System')
    parser.add_argument('--source', type=str, required=True,
                        help='Video source (file path or camera index)')
    parser.add_argument('--output', type=str,
                        help='Output video path (optional)')
    parser.add_argument('--mark-zones', action='store_true',
                        help='Start in zone marking mode')

    args = parser.parse_args()

    monitor = RestrictedZoneMonitor()

    if args.mark_zones:
        print("Starting zone marking mode...")
        monitor.mark_zones_interactively(args.source)
    else:
        print("Starting monitoring mode...")
        monitor.process_video(args.source, args.output)


if __name__ == "__main__":
    main()
