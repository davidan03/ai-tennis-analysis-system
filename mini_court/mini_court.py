import cv2
import constants
from utils import convert_pixels_to_meters, convert_meters_to_pixels
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