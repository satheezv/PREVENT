import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from torchvision.models import efficientnet_b0
import os

# ✅ Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Define dataset directory (to get class names)
dataset_dir = "W:/gasSensor_ws/GasSensor_ws/data_v2/core"

# ✅ Load class mappings
class_folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
class_map = {idx: class_name for idx, class_name in enumerate(sorted(class_folders))}

# ✅ Define image preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ Load Trained Model
model = efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_map))
model.load_state_dict(torch.load("best_efficientnet.pth", map_location=device))
model.eval().to(device)

# ✅ Start Video Capture
cap = cv2.VideoCapture(0)  # 0 for default webcam, or replace with video file path

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# ✅ Real-Time Video Inference
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    # Convert OpenCV BGR frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)

    # Preprocess frame
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Run prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_label = class_map[predicted.item()]

    # Display prediction on frame
    cv2.putText(frame, f"Prediction: {predicted_label}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show video feed
    cv2.imshow("Live Classification", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Release resources
cap.release()
cv2.destroyAllWindows()
