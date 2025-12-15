from ultralytics import YOLO
import cv2
import pickle

class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

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