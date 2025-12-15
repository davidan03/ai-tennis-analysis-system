from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector

def main():
    # Declare video path and read video frames
    input_video_path = "inputs/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Create PlayerTracker object using pre-trained yolo11x and
    # retrieve the player detections from the video (list of
    # ids to bounding box coords dictionaries)
    player_tracker = PlayerTracker("models/yolo11x.pt")

    # Create BallTracker object using fine-tuned yolo11x model trained on
    # Roboflow dataset and retrieve the ball detections from the video
    # (list of singular id to bounding box coords dictionaries)
    ball_tracker = BallTracker("models/yolo11x_best_tennis_ball_detector.pt")

    # Create CourtLineDetector object using the trained CNN
    court_line_detector = CourtLineDetector("models/keypoints_model.pth")

    player_detections = player_tracker.detect_frames(video_frames,
                                                    read_from_stub=True,
                                                     stub_path="tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                 read_from_stub=True,
                                                 stub_path="tracker_stubs/ball_detections.pkl")
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Draw bounding boxes on the video frames
    output_video_frames = player_tracker.draw_bounding_boxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bounding_boxes(video_frames, ball_detections)

    # Predict court keypoints and draw them onto the video frames
    court_keypoints = court_line_detector.predict(video_frames[0])
    output_video_frames = court_line_detector.draw_keypoints_on_video(video_frames, court_keypoints)

    output_video_path = "outputs/output_video.avi"
    save_video(output_video_frames, output_video_path)

if __name__ == "__main__":
    main()