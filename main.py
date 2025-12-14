from utils import read_video, save_video
from trackers import PlayerTracker

def main():
    # Declare video path and read video frames
    input_video_path = "inputs/input_video.mp4"
    video_frames = read_video(input_video_path)

    # Create PlayerTracker object using pre-trained yolo11x and
    # retrieve the player detections from the video (list of
    # ids to bounding box coords dictionaries)
    player_tracker = PlayerTracker("models/yolo11x.pt")
    player_detections = player_tracker.detect_frames(video_frames)

    # Draw bounding boxes on the video frames
    output_video_frames = player_tracker.draw_bounding_boxes(video_frames, player_detections)

    output_video_path = "outputs/output_video.avi"
    save_video(output_video_frames, output_video_path)

if __name__ == "__main__":
    main()