import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import os

# ✅ Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Define dataset directory (to get class names)
dataset_dir = "W:/gasSensor_ws/GasSensor_ws/data_v2/unbalanced"

# ✅ Load class mappings
class_folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
class_map = {idx: class_name for idx, class_name in enumerate(sorted(class_folders))}

# ✅ Define image preprocessing (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ Load Trained Model
model = efficientnet_b0(weights=None)  # No pretrained weights, using trained model
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_map))
model.load_state_dict(torch.load("best_efficientnet_final.pth", map_location=device))
model.eval().to(device)

# ✅ Function to classify an input image
def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Load image
    input_tensor = transform(image).unsqueeze(0).to(device)  # Preprocess and add batch dimension

    # ✅ Run model prediction
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_label = class_map[predicted.item()]

    print(f"✅ Predicted Class: {predicted_label}, predicted item: {predicted.item()}")

# ✅ Test the model with an image
if __name__ == "__main__":
    test_image_path = "W:\gasSensor_ws\GasSensor_ws\data_v2\core\perfect_rack\image_0002.jpg"  # Change this to your test image path
    classify_image(test_image_path)
