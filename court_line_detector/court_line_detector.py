import torch
from torchvision import transforms, models
import cv2

class CourtLineDetector:
    def __init__(self, model_path):
        # Have to set up correct model architecture before loading parameters
        self.model = models.resnet50(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14*2)

        # state_dict is a Python dictionary that maps each layer to its
        # parameter tensor
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))

        self.transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def predict(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Transforms returns a PyTorch tensor
        # Uses unsqueeze(0) to add a batch dimension as PyTorch
        # models expect a batched input
        image_tensor = self.transforms(image_rgb).unsqueeze(0)

        # Disables gradient computation and saves memory
        with torch.no_grad():
            outputs = self.model(image_tensor)

        # Removes batch dimension, moves tensor to CPU (required for NumPy conversion),
        # then converts to NumPy array
        keypoints = outputs.squeeze().cpu().numpy()

        # Scale keypoints from [0, 1] back to original image dimensions
        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w
        keypoints[1::2] *= original_h

        return keypoints
    
    def draw_keypoints(self, image, keypoints):
        for i in range(0, len(keypoints), 2):
            # Need to wrap in int() because the NumPy array contains floats
            # The CNN works with real values, so they will be floats
            x = int(keypoints[i])
            y = int(keypoints[i + 1])
            
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(image, str(int(i / 2)), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return image
    
    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []

        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)

        return output_video_frames
