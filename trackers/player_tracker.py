from ultralytics import YOLO
import cv2
import pickle
from utils import get_center_of_box, distance_between_points

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def filter_players(self, court_keypoints, player_detections):
        """
        Returns a list of dictionaries of player IDs to bounding box coordinates
        of the two actual players. Umpire, line judges, and ball children were
        incorrectly identified as players, making this filtering necessary.
        
        :param court_keypoints: list of court keypoint coordinates
        :param player_detections: list of dictionaries of player IDs to bounding
                                  box coordinates (x min, y min) -> (x max, y max)
        """
        # Takes the first dictionary, equating to the detections from the first
        # video frame and makes a call to the helper method
        player_detections_first_frame = player_detections[0]
        filtered_players = self.filter_players_helper(court_keypoints, player_detections_first_frame)
        filtered_player_detections = []

        # Uses list comprehension to filter for only the dictionaries of
        # the actual players
        for player_dict in player_detections:
            filtered_player_dict = {id: bounding_box for id, bounding_box in player_dict.items() if id in filtered_players}
            filtered_player_detections.append(filtered_player_dict)

        return filtered_player_detections
    
    def filter_players_helper(self, court_keypoints, player_detection):
        """
        Generates a list of player IDs to their closest distance to a court keypoint.
        Returns a list of IDs of the two closest players to the court, based on the
        given keypoints.
        
        :param court_keypoints: list of court keypoint coordinates
        :param player_detection: dictionary of player IDs to bounding box coordinates
                                 (x min, y min) -> (x max, y max)
        """
        distances = []

        # player_detection is a dictionary representing one video frame's worth of detections
        # Maps player IDs to their bounding box coordinates
        for id, bounding_box in player_detection.items():
            player_center = get_center_of_box(bounding_box)

            min_distance = float("inf")

            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i + 1])
                distance = distance_between_points(court_keypoint, player_center)

                # Determines the closest distance between the player and the court
                if distance < min_distance:
                    min_distance = distance
            
            # Appends the closest distance
            distances.append((id, min_distance))

        # Sorts the (ID, min_distance) tuples by ascending min_distance
        # Filters the IDs of the two players with the closest distance to the court
        distances.sort(key=lambda x: x[1])
        filtered_players = [distances[0][0], distances[1][0]]
        
        return filtered_players

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """
        Detects players in a list of video frames.
        
        :param frames: list of NumPy arrays representing video frames
        :param read_from_stub: bool indicating whether to read detections from a stub file
        :param stub_path: path to the stub file for reading/writing detections
        :return: list of dictionaries mapping player IDs to bounding box coordinates
        """
        player_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                player_detections = pickle.load(f)
            
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        """
        Detects various players in a single video frame and assigns them
        unique IDs.

        :param frame: a NumPy array representing a single video frame
        """
        # Need to index 0 because track() returns a list of Result objects, one per input image
        results = self.model.track(frame, persist=True)[0]

        # Results.names is a dictionary mapping class IDs to class names
        id_name_dict = results.names

        player_dict = {}

        for box in results.boxes:
            # Use .tolist()[0] to convert from tensor to native Python type
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]

            if object_cls_name == "person":
                player_dict[track_id] = result

        return player_dict
    
    def draw_bounding_boxes(self, video_frames, player_detections):
        """
        Draws blue bounding boxes around players and annotates them
        with green text that is buffered vertically.
        
        :param video_frames: list of NumPy arrays representing video frames
        :param player_detections: list of dictionaries mapping player IDs to bounding box coordinates
        """
        output_video_frames = []

        for frame, player_dict in zip(video_frames, player_detections):
            for id, bounding_box in player_dict.items():
                # Coordinates represent x min, y min, x max, y max
                x1, y1, x2, y2 = bounding_box
                cv2.putText(frame, f"Player ID: {id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            
            output_video_frames.append(frame)
        
        return output_video_frames