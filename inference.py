from ultralytics import YOLO

# Testing out the YOLO model and specific functionality
model = YOLO("./training/yolo11x_best_tennis_ball_detector.pt")

result = model.track("./inputs/input_video.mp4",
                       save=True,
                       project="/Users/davidan/ai-tennis-analysis-system/ai-tennis-analysis-system/runs/detect")

print(result)
print("boxes:")

for box in result[0].boxes:
    print(box)