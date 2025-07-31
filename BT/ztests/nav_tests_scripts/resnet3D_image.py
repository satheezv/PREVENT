import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18
from PIL import Image
import argparse

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

# ✅ Define image transformation
transform = transforms.Compose([
    transforms.Resize((112, 112)),  # Resize to model input size
    transforms.ToTensor(),
])

# ✅ Function to classify an input image
def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Load and convert to RGB
    input_tensor = transform(image).unsqueeze(0).unsqueeze(2)  # Reshape to (1, C, T, H, W)
    input_tensor = input_tensor.to(device)

    # ✅ Run model prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        print("Raw Outputs:", outputs)  # Debugging line
        _, predicted = torch.max(outputs, 1)
        print("Prediction", predicted.item())
        predicted_label = class_map[predicted.item()]

    print(f"✅ Predicted Class: {predicted_label}")

# ✅ Main function to run inference
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Test ResNet3D on an input image")
    # parser.add_argument("image_path", type=str, help="Path to input image")
    # args = parser.parse_args()
    path = "W:\gasSensor_ws\GasSensor_ws\data_v2\\total\\rack_missing_vials\image_0023_fx.jpg"
    classify_image(path)
