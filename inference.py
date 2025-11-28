from ultralytics import YOLO

model = YOLO("./training/best.pt")

result = model.track("./inputs/input_video.mp4",
                       save=True,
                       project="/Users/davidan/ai-tennis-analysis-system/ai-tennis-analysis-system/runs/detect")

print(result)
print("boxes:")

for box in result[0].boxes:
    print(box)