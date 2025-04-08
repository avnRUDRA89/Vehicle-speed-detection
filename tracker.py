import math


class Tracker:
    def __init__(self):
        # Store tracks for each object (consider using a Track class for better organization)
        # Each track is a dictionary with 'id', 'center', and optionally 'last_seen' (for timeout)
        self.tracks = {}
        self.id_count = 0  # Keep a count for assigning unique IDs

    def update(self, objects_rect, max_distance=35, max_inactive=10):  # Allow customization of thresholds
        """
        Updates the tracker with new detections from the current frame.

        Args:
            objects_rect (list): A list of bounding boxes representing detected objects.
                Each bounding box can be a list or dictionary containing coordinates (x, y, w, h).
            max_distance (int, optional): The maximum distance between center points
                to consider an object as the same (default: 35).
            max_inactive (int, optional): The maximum number of frames an object can be
                inactive before being removed (default: 10).

        Returns:
            list: A list of objects with bounding boxes and IDs (dictionaries).
        """

        updated_objects = []

        # Get center points of new objects
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find matching existing track using nearest neighbor
            closest_track = None
            closest_distance = math.inf  # Initialize with positive infinity

            for track_id, track in self.tracks.items():
                center_x, center_y = track['center']
                dist = math.hypot(cx - center_x, cy - center_y)  # Euclidean distance

                if dist < closest_distance:
                    closest_track = track_id
                    closest_distance = dist

            # Match if distance is less than threshold
            if closest_distance <= max_distance:
                self.tracks[closest_track]['center'] = (cx, cy)  # Update center
                updated_objects.append({'bbox': rect, 'id': closest_track})

            # New object detected, assign a new ID
            else:
                self.id_count += 1
                self.tracks[self.id_count] = {'center': (cx, cy), 'id': self.id_count}
                updated_objects.append({'bbox': rect, 'id': self.id_count})

        # Clean inactive tracks (objects not seen for a while)
        inactive_tracks = []
        for track_id, track in self.tracks.items():
            if 'last_seen' not in track:  # Add 'last_seen' on first detection
                track['last_seen'] = 0
            track['last_seen'] += 1
            if track['last_seen'] > max_inactive:
                inactive_tracks.append(track_id)

        for id in inactive_tracks:
            del self.tracks[id]

        return updated_objects
