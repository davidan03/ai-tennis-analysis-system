import cv2
import constants
from utils import (convert_pixels_to_meters,
                   convert_meters_to_pixels,
                   get_foot_position,
                   get_closest_keypoint_index,
                   get_bounding_box_height,
                   measure_xy_distance,
                   get_center_of_box,
                   distance_between_points)
import numpy as np

class MiniCourt:
    def __init__(self, frame):
        self.rectangle_width = 250
        self.rectangle_length = 450
        self.buffer = 50 # Buffer between rectangle and the edge of the video
        self.padding = 20 # Padding between rectangle and the mini court

        self.set_rectangle_dimensions(frame)
        self.set_mini_court_dimensions()
        self.set_mini_court_keypoints()
        self.set_mini_court_lines()

    def convert_meters_to_pixels(self, meters):
        """
        Returns pixels from meters.
        
        :param meters: distance to be converted to pixels
        """
        return convert_meters_to_pixels(meters, self.court_width, constants.DOUBLES_LINE_WIDTH)

    def set_rectangle_dimensions(self, frame):
        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.rectangle_length
        self.start_x = self.end_x - self.rectangle_width
        self.start_y = self.end_y - self.rectangle_length

    def set_mini_court_dimensions(self):
        self.court_start_x = self.start_x + self.padding
        self.court_start_y = self.start_y + self.padding
        self.court_end_x = self.end_x - self.padding
        self.court_end_y = self.end_y - self.padding
        self.court_width = self.court_end_x - self.court_start_x

    def set_mini_court_keypoints(self):
        keypoints = [0] * 28 # 14 keypoints with x and y coordinates

        # Keypoints are defined from the camera's perspective
        # point 0 (far left corner doubles line)
        keypoints[0], keypoints[1] = int(self.court_start_x), int(self.court_start_y)

        # point 1 (far right corner doubles line)
        keypoints[2], keypoints[3] = int(self.court_end_x), int(self.court_start_y)

        # point 2 (close left corner doubles line)
        keypoints[4], keypoints[5] = int(self.court_start_x), int(self.court_end_y)

        # point 3 (close right corner doubles line)
        keypoints[6], keypoints[7] = int(self.court_end_x), int(self.court_end_y)

        # point 4 (far left corner singles line)
        keypoints[8] = int(self.court_start_x + self.convert_meters_to_pixels(constants.DOUBLES_ALLEY_DIFF))
        keypoints[9] = int(self.court_start_y)

        # point 5 (far right corner singles line)
        keypoints[10] = int(self.court_end_x - self.convert_meters_to_pixels(constants.DOUBLES_ALLEY_DIFF))
        keypoints[11] = int(self.court_start_y)

        # point 6 (close left corner singles line)
        keypoints[12] = int(self.court_start_x + self.convert_meters_to_pixels(constants.DOUBLES_ALLEY_DIFF))
        keypoints[13] = int(self.court_end_y)

        # point 7 (close right corner singles line)
        keypoints[14] = int(self.court_end_x - self.convert_meters_to_pixels(constants.DOUBLES_ALLEY_DIFF))
        keypoints[15] = int(self.court_end_y)

        # point 8 (far left corner service line)
        keypoints[16] = int(self.court_start_x + self.convert_meters_to_pixels(constants.DOUBLES_ALLEY_DIFF))
        keypoints[17] = int(self.court_start_y + self.convert_meters_to_pixels(constants.NO_MANS_LAND_WIDTH))

        # point 9 (far right corner service line)
        keypoints[18] = int(self.court_end_x - self.convert_meters_to_pixels(constants.DOUBLES_ALLEY_DIFF))
        keypoints[19] = int(self.court_start_y + self.convert_meters_to_pixels(constants.NO_MANS_LAND_WIDTH))

        # point 10 (close left corner service line)
        keypoints[20] = int(self.court_start_x + self.convert_meters_to_pixels(constants.DOUBLES_ALLEY_DIFF))
        keypoints[21] = int(self.court_end_y - self.convert_meters_to_pixels(constants.NO_MANS_LAND_WIDTH))

        # point 11 (close right corner service line)
        keypoints[22] = int(self.court_end_x - self.convert_meters_to_pixels(constants.DOUBLES_ALLEY_DIFF))
        keypoints[23] = int(self.court_end_y - self.convert_meters_to_pixels(constants.NO_MANS_LAND_WIDTH))

        # point 12 (far down the t)
        keypoints[24] = int(keypoints[16] + self.convert_meters_to_pixels(constants.SINGLES_LINE_WIDTH) / 2)
        keypoints[25] = int(self.court_start_y + self.convert_meters_to_pixels(constants.NO_MANS_LAND_WIDTH))

        # point 13 (close down the t)
        keypoints[26] = keypoints[24]
        keypoints[27] = int(self.court_end_y - self.convert_meters_to_pixels(constants.NO_MANS_LAND_WIDTH))

        self.keypoints = keypoints
    
    def set_mini_court_lines(self):
        # Numbers represent the label of the keypoint in the output video
        self.lines = [
            (0, 1),
            (0, 2),
            (1, 3),
            (2, 3),
            (4, 6),
            (4, 5),
            (5, 7),
            (6, 7),
            (10, 11),
            (8, 9)
        ]

    def draw_background_rectangle(self, frame):
        # Create a NumPy array with same shape as frame
        # but all values are 0
        shapes = np.zeros_like(frame, np.uint8)

        # Draw the background rectangle onto shapes (-1 means filled)
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y),
                      (255, 255, 255), -1)
        
        # Create a deep copy of frame to avoid modifying it
        output = frame.copy()
        alpha = 0.5 # Equivalent to transparency

        # 0 == False
        # Anything non-zero == True
        mask = shapes.astype(bool)

        # Blends two images by calculating their weighted sum, creating transparency effect
        # Any position where mask is True (inside of the rectangle), the transparency effect
        # is added
        output[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return output
    
    def draw_mini_court_features(self, frame):
        # Draw the keypoints as filled dots
        for i in range(0, len(self.keypoints), 2):
            x = int(self.keypoints[i])
            y = int(self.keypoints[i + 1])
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        # Draw lines
        for line in self.lines:
            start_point = (int(self.keypoints[line[0] * 2]), int(self.keypoints[line[0] * 2 + 1]))
            end_point = (int(self.keypoints[line[1] * 2]), int(self.keypoints[line[1] * 2 + 1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Draw net
        net_start_point = (int(self.court_start_x), int(self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LENGTH)) - 20)
        net_end_point = (int(self.court_end_x), int(self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LENGTH)) - 20)
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame
    
    def draw_mini_court(self, frames):
        output_video_frames = []

        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_mini_court_features(frame)
            output_video_frames.append(frame)

        return output_video_frames
    
    def get_mini_court_start_point(self):
        return (self.court_start_x, self.court_start_y)
    
    def get_mini_court_width(self):
        return self.court_width
    
    def get_mini_court_keypoints(self):
        return self.keypoints
    
    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, court_keypoints):
        player_heights = {
            1: constants.PLAYER_1_HEIGHT,
            2: constants.PLAYER_2_HEIGHT
        }

        # List of player IDs to mini court position dictionaries
        output_player_boxes = []

        # List of ball ID to mini court position dictionaries
        output_ball_boxes = []

        for frame_num, player_dict in enumerate(player_boxes):
            ball_box = ball_boxes[frame_num][1]
            ball_position = get_center_of_box(ball_box)
            
            # Gets player ID of closest player to the ball
            # Iterates over the keys (player IDs) and gets the ID with
            # the minimum distance by comparing ball_position to the
            # center of the player's bounding box
            closest_player_id_to_ball = min(player_dict.keys(),
                                            key=lambda x:
                                            distance_between_points(
                                                ball_position,
                                                get_center_of_box(player_dict[x])
                                                ))

            # Dict of player IDs to mini court position
            output_player_bounding_box_dict = {}
            output_ball_bounding_box_dict = {}

            for player_id, bounding_box in player_dict.items():
                foot_position = get_foot_position(bounding_box)

                # Determines closest keypoint to the player's feet
                closest_keypoint_index = get_closest_keypoint_index(foot_position, court_keypoints, [0, 2, 12, 13])
                closest_keypoint = (court_keypoints[closest_keypoint_index * 2],
                                    court_keypoints[closest_keypoint_index * 2 + 1])

                # Get the player's height in pixels
                # Min goes 20 frames before if possible
                # Max goes 50 frames after if possible
                frame_num_min = max(0, frame_num - 20)
                frame_num_max = min(len(player_boxes), frame_num + 50)

                # Retrieves the heights of the player with player_id for the frame range
                # There are various heights of the player because of the camera angle and how they
                # move during the match, lunges etc.
                bounding_box_heights_in_pixels = [get_bounding_box_height(player_boxes[i][player_id])
                                                  for i in range(frame_num_min, frame_num_max)]
                
                # Max height from frame range should be the player's actual height
                # (standing straight up height)
                player_height_in_pixels = max(bounding_box_heights_in_pixels)

                # Get player's mini court position
                mini_court_player_pos = self.get_mini_court_coordinates(foot_position,
                                                                        closest_keypoint,
                                                                        closest_keypoint_index,
                                                                        player_height_in_pixels,
                                                                        player_heights[player_id])
                
                output_player_bounding_box_dict[player_id] = mini_court_player_pos

                if closest_player_id_to_ball == player_id:
                    # Get closest keypoint in pixels
                    closest_keypoint_index = get_closest_keypoint_index(ball_position, court_keypoints, [0, 2, 12, 13])
                    closest_keypoint = (court_keypoints[closest_keypoint_index * 2],
                                        court_keypoints[closest_keypoint_index * 2 + 1])
                    
                    # Get ball's mini court position
                    mini_court_ball_pos = self.get_mini_court_coordinates(ball_position,
                                                                        closest_keypoint,
                                                                        closest_keypoint_index,
                                                                        player_height_in_pixels,
                                                                        player_heights[player_id])
                    
                    output_ball_boxes.append({1: mini_court_ball_pos})
                
            output_player_boxes.append(output_player_bounding_box_dict)

        return output_player_boxes, output_ball_boxes

    def get_mini_court_coordinates(self, player_position,
                                   closest_keypoint, closest_keypoint_index,
                                   player_height_in_pixels, player_height_in_meters):
        """
        Returns the mini court position of the player.
        
        :param player_position: the player's actual position
        :param closest_keypoint: the actual keypoint closest to the player
        :param closest_keypoint_index: the label of the keypoint
        :param player_height_in_pixels: player's height in pixels
        :param player_height_in_meters: player's height in meters
        """
        # Distance between individual x and y coordinates
        # Returned as an (x, y) point
        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(player_position, closest_keypoint)

        # Convert pixels to meters
        distance_from_keypoint_x_meters = convert_pixels_to_meters(distance_from_keypoint_x_pixels,
                                                                   player_height_in_meters,
                                                                   player_height_in_pixels)
        distance_from_keypoint_y_meters = convert_pixels_to_meters(distance_from_keypoint_y_pixels,
                                                                   player_height_in_meters,
                                                                   player_height_in_pixels)
        
        # Normalize to mini court coordinates
        mini_court_x_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_x_meters)
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_y_meters)

        # Convert closest keypoint to mini court coords
        closest_mini_court_keypoint = (self.keypoints[closest_keypoint_index * 2],
                                       self.keypoints[closest_keypoint_index * 2 + 1])
        
        mini_court_player_pos = (closest_mini_court_keypoint[0] + mini_court_x_distance_pixels,
                                 closest_mini_court_keypoint[1] + mini_court_y_distance_pixels)
        
        return mini_court_player_pos
    
    def draw_points_on_mini_court(self, frames, positions, color=(0, 255, 0)):
        for frame_num, frame in enumerate(frames):
            for _, position in positions[frame_num].items():
                x, y = position
                x = int(x)
                y = int(y)
                cv2.circle(frame, (x, y), 5, color, -1)
        
        return frames