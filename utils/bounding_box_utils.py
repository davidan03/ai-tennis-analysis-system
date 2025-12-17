import math

def get_center_of_box(bounding_box):
    x1, y1, x2, y2 = bounding_box
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    
    return (center_x, center_y)

def distance_between_points(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    distance = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))

    return distance

def measure_xy_distance(p1, p2):
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])

def get_foot_position(player_bounding_box):
    x1, y1, x2, y2 = player_bounding_box

    return (int((x1 + x2) / 2), y2)

def get_closest_keypoint_index(point, court_keypoints, keypoint_indices):
    closest_distance = float("inf")

    # Candidate for closest keypoint index
    keypoint_index = keypoint_indices[0]

    for index in keypoint_indices:
        # Retrieve x and y coordinates
        keypoint = court_keypoints[index * 2], court_keypoints[index * 2 + 1]
        distance = abs(point[1] - keypoint[1])

        if distance < closest_distance:
            closest_distance = distance
            keypoint_index = index

    return keypoint_index

def get_bounding_box_height(bounding_box):
    return bounding_box[3] - bounding_box[1]