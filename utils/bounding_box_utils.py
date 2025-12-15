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