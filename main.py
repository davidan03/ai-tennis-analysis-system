from utils import read_video, save_video, distance_between_points, convert_pixels_to_meters, draw_player_stats
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
import cv2
from mini_court import MiniCourt
import constants
from copy import deepcopy
import pandas as pd

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
    
    # Get frames where ball was hit (as a list of frame numbers)
    ball_hit_frames = ball_tracker.get_ball_hit_frames(ball_detections)

    # Establish stats we want to display
    player_stats_data = [{
        'frame_num': 0,

        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,

        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0,
    }]

    # Calculate the speed of the opponent during a shot and
    # the speed of the ball itself
    # We don't account for the last shot because we need a shot
    # following it to determine speed
    for index in range(len(ball_hit_frames) - 1):
        start_frame = ball_hit_frames[index]
        end_frame = ball_hit_frames[index + 1]
        ball_shot_time_seconds = (end_frame - start_frame) / 24 # The video is 24 fps

        # Get distance covered by ball
        ball_distance_covered_pixels = distance_between_points(ball_mini_court_detections[start_frame][1],
                                                               ball_mini_court_detections[end_frame][1])
        ball_distance_covered_meters = convert_pixels_to_meters(ball_distance_covered_pixels,
                                                                constants.DOUBLES_LINE_WIDTH,
                                                                mini_court.get_mini_court_width())
        
        # Speed of the ball shot in km/h
        speed_of_ball_shot = ball_distance_covered_meters / ball_shot_time_seconds * 3.6

        # Player who shot the ball
        player_dict = player_mini_court_detections[start_frame]
        player_shot_ball = min(player_dict.keys(), key=lambda id: distance_between_points(player_dict[id],
                                                                                          ball_mini_court_detections[start_frame][1]))
        
        # Opponent player
        opponent_player_id = 1 if player_shot_ball == 2 else 2

        # Opponent player speed
        distance_covered_by_opponent_pixels = distance_between_points(player_mini_court_detections[start_frame][opponent_player_id],
                                                                      player_mini_court_detections[end_frame][opponent_player_id])
        distance_covered_by_opponent_meters = convert_pixels_to_meters(distance_covered_by_opponent_pixels,
                                                                       constants.DOUBLES_LINE_WIDTH,
                                                                       mini_court.get_mini_court_width())
        opponent_player_speed = distance_covered_by_opponent_meters / ball_shot_time_seconds * 3.6

        # Make a deepcopy to only copy over values from dict
        # Index -1 because we are accumulating stats, so we
        # want to build off the previous state
        current_player_stats = deepcopy(player_stats_data[-1])

        # Update current player's (the player who just shot the ball) stats
        current_player_stats["frame_num"] = start_frame
        current_player_stats[f"player_{player_shot_ball}_number_of_shots"] += 1
        current_player_stats[f"player_{player_shot_ball}_total_shot_speed"] += speed_of_ball_shot
        current_player_stats[f"player_{player_shot_ball}_last_shot_speed"] = speed_of_ball_shot

        # Update opponent's stats
        current_player_stats[f"player_{opponent_player_id}_total_player_speed"] += opponent_player_speed
        current_player_stats[f"player_{opponent_player_id}_last_player_speed"] = opponent_player_speed

        # Add to player_stats_data to update the list
        player_stats_data.append(current_player_stats)

    # Convert player_stats_data into a dataframe
    # Rows only exist where there was a hit
    df_player_stats_data = pd.DataFrame(player_stats_data)

    # Create a frames dataframe with a row per frame
    df_frames = pd.DataFrame({"frame_num": list(range(len(video_frames)))})

    # df_frames is the left table
    # df_player_stats_data is the right table
    # how="left" tells pandas to preserve key order of left table
    # on="frame_num" tells pandas that stats should appear where frame_num matches
    df_player_stats_data = pd.merge(df_frames, df_player_stats_data, how="left", on="frame_num")

    # Replaces NaN (frames where a hit was not detected)
    # with the last known valid value
    df_player_stats_data = df_player_stats_data.ffill()

    # Calculate average shot speed and average player speed
    # for both players
    df_player_stats_data["player_1_average_shot_speed"] = df_player_stats_data["player_1_total_shot_speed"] / df_player_stats_data["player_1_number_of_shots"]
    df_player_stats_data["player_2_average_shot_speed"] = df_player_stats_data["player_2_total_shot_speed"] / df_player_stats_data["player_2_number_of_shots"]
    df_player_stats_data["player_1_average_player_speed"] = df_player_stats_data["player_1_total_player_speed"] / df_player_stats_data["player_2_number_of_shots"]
    df_player_stats_data["player_2_average_player_speed"] = df_player_stats_data["player_2_total_player_speed"] / df_player_stats_data["player_1_number_of_shots"]

    # Draw bounding boxes on the video frames
    output_video_frames = player_tracker.draw_bounding_boxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bounding_boxes(video_frames, ball_detections)

    # Draw mini court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)

    # Draw real-time player movement on mini court
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections, color=(255, 0, 0))

    # Draw real-time ball movement on mini court
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections)

    # Draw player stats on video frames
    output_video_frames = draw_player_stats(output_video_frames, df_player_stats_data)

    # Write frame number in top left corner for each frame
    # Helps with debugging and figuring out where in the video we are
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    output_video_path = "outputs/output_video.avi"
    save_video(output_video_frames, output_video_path)

if __name__ == "__main__":
    main()