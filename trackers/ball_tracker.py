from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_positions(self, ball_positions):
        # Get bounding box coordinates otherwise empty list
        ball_positions = [x.get(1, []) for x in ball_positions]

        # Convert list into a pandas dataframe
        # Empty lists become NaN
        df_ball_positions = pd.DataFrame(ball_positions, columns=["x1", "y1", "x2", "y2"])

        # Interpolate missing values and backfill where start
        # of sequence is missing (unable to interpolate)
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()]
    
        return ball_positions

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """
        Detects tennis balls in a list of video frames.
        
        :param frames: list of NumPy arrays representing video frames
        :param read_from_stub: bool indicating whether to read detections from a stub file
        :param stub_path: path to the stub file for reading/writing detections
        :return: list of dictionaries mapping the ball ID to bounding box coordinates
        """
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, "rb") as f:
                ball_detections = pickle.load(f)
            
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        if stub_path is not None:
            with open(stub_path, "wb") as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame):
        """
        Detects tennis ball in a single video frame.
        
        :param frame: a NumPy array representing a single video frame
        """

        # Need to index 0 because track() returns a list of Result objects, one per input image
        # conf sets the minimum confidence threshold for detection
        results = self.model.predict(frame, conf=0.15)[0]
        ball_dict = {}

        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result

        return ball_dict
    
    def draw_bounding_boxes(self, video_frames, ball_detections):
        """
        Draws red bounding boxes around the tennis ball and annotates them
        with green text that is buffered vertically.
        
        :param video_frames: list of NumPy arrays representing video frames
        :param ball_detections: list of dictionaries mapping the ball ID to bounding box coordinates
        """

        output_video_frames = []

        for frame, ball_dict in zip(video_frames, ball_detections):
            for id, bounding_box in ball_dict.items():
                # Coordinates represent x min, y min, x max, y max
                x1, y1, x2, y2 = bounding_box
                cv2.putText(frame, f"Ball ID: {id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            
            output_video_frames.append(frame)
        
        return output_video_frames