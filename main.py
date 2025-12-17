from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import cv2
from mini_court import MiniCourt

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

    # Create MiniCourt object to draw the real-time mini court in the top right of the video
    mini_court = MiniCourt(video_frames[0])

    # Retrieve list of dictionaries of player IDs to bounding box coordinates
    player_detections = player_tracker.detect_frames(video_frames,
                                                    read_from_stub=True,
                                                     stub_path="tracker_stubs/player_detections.pkl")
    
    # Retrieve list of dictionaries of the ball's ID to bounding box coordinates
    ball_detections = ball_tracker.detect_frames(video_frames,
                                                 read_from_stub=True,
                                                 stub_path="tracker_stubs/ball_detections.pkl")
    
    # Interpolate ball positions where detections don't occur
    ball_detections = ball_tracker.interpolate_ball_positions(ball_detections)

    # Predict court keypoints and draw them onto the video frames
    court_keypoints = court_line_detector.predict(video_frames[0])
    output_video_frames = court_line_detector.draw_keypoints_on_video(video_frames, court_keypoints)

    # Filter for only the two actual players
    player_detections = player_tracker.filter_players(court_keypoints, player_detections)

    # Convert positions to mini court positions
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(player_detections,
                                                                                                                           ball_detections,
                                                                                                                           court_keypoints)

    # Draw bounding boxes on the video frames
    output_video_frames = player_tracker.draw_bounding_boxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bounding_boxes(video_frames, ball_detections)

    # Draw mini court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)

    # Draw real-time player movement on mini court
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections, color=(255, 0, 0))

    # Draw real-time ball movement on mini court
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections)

    # Write frame number in top left corner for each frame
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    output_video_path = "outputs/output_video.avi"
    save_video(output_video_frames, output_video_path)

if __name__ == "__main__":
    main()