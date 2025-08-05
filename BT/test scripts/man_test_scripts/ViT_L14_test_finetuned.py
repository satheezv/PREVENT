import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import os

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the fine-tuned model
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
model.load_state_dict(torch.load("W:\gasSensor_ws\GasSensor_ws\src\BT\ztests\man_test_scripts\clip_vit_l14_nav_hazard_classification.pth", map_location=device),strict=False)
model.eval()

# Load the processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Class labels
# class_labels = ["Hazard", "No Hazard"]

class_labels = ["empty floor", "object on the floor"]

# Prediction function
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")

    # Preprocess the image and text
    inputs = processor(
        text=class_labels,           # Class labels as text
        images=image,                # Input image
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image  # Extract logits for classification

    # Prediction
    predicted_class_idx = logits.argmax().item()
    predicted_label = class_labels[predicted_class_idx]

    print(f"🖼️ Image: {os.path.basename(image_path)} → Prediction: {predicted_label}")
    return predicted_label

# Batch Prediction (Optional)
def predict_folder(folder_path):
    results = {}
    for image_file in os.listdir(folder_path):
        if image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            image_path = os.path.join(folder_path, image_file)
            prediction = predict_image(image_path)
            results[image_file] = prediction

    print("\n📋 Prediction Summary:")
    for img, label in results.items():
        print(f"{img}: {label}")

# Example Usage
# image_path = "path_to_your_test_image.jpg"
# predict_image(image_path)

# For folder-based prediction
# no_hazard_folder_path = "W:\gasSensor_ws\others\data for resnet\man_dataset\\test\exp1 - no hazard 25 runs 13_3_25"
# Hazard_folder_path = "W:\gasSensor_ws\others\data for resnet\man_dataset\\test\exp2 - hazard 25 runs 13_3_25"
empty_floor_path = "W:\gasSensor_ws\others\\nav experiment obs images\\vlm test\empty floor"
# obstacle_path = "W:\gasSensor_ws\others\\nav experiment obs images\\vlm test\object on the floor"
predict_folder(empty_floor_path)
