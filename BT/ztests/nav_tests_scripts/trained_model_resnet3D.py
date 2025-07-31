import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from collections import deque
from torchvision.models.video import r3d_18
from PIL import Image

# ✅ Load trained model
model_path = "resnet3d_trained_v2.pth"
device = torch.device("cpu")  # Use CPU since you don't have CUDA

# ✅ Define class labels (same as training)
class_map = {
    0: "perfect_rack",
    1: "rack_missing_vials"
}  # Adjust based on your training classes

# ✅ Load ResNet3D model
model = r3d_18(weights=None)  # No pre-trained weights, using fine-tuned model
model.fc = nn.Linear(model.fc.in_features, len(class_map))  # Adjust output layer
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# ✅ Define transformations
transform = transforms.Compose([
    transforms.Resize((112, 112)),  # Resize to model input size
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

# ✅ Set up video capture
cap = cv2.VideoCapture(0)  # Use 0 for default webcam, change if needed
sequence_length = 3  # Same as training
frame_buffer = deque(maxlen=sequence_length)  # Store last `sequence_length` frames

print("📸 Starting live classification... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture image")
        break

    # Convert frame to PIL image and apply transformations
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    processed_frame = transform(pil_image)

    # Add frame to buffer
    frame_buffer.append(processed_frame)

    # When enough frames are collected, make a prediction
    if len(frame_buffer) == sequence_length:
        input_tensor = torch.stack(list(frame_buffer), dim=1).unsqueeze(0)  # Shape: (1, C, T, H, W)
        input_tensor = input_tensor.to(device)

        # ✅ Run model prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_label = class_map[predicted.item()]

        # ✅ Display result
        cv2.putText(frame, f"Prediction: {predicted_label}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ✅ Show live feed
    cv2.imshow("Live Camera - ResNet3D Classification", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Cleanup
cap.release()
cv2.destroyAllWindows()
print("✅ Live classification ended.")
