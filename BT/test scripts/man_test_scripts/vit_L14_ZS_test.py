import torch
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F
from PIL import Image
import os

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pretrained CLIP model (without fine-tuned weights)
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
model.eval()

# Load the processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Class labels

#T1
# class_labels = ["empty floor", "Tissue paper", "glass piece", "vial cap", "liquid spillage", "solvent bottle", "purple glove", "screwdriver", "goggle" ]

# T2 
class_labels = ["vial missing a black cap", "rack with empty round slot", "vials in a rack with no empty slot", "liquid spillage", "a purple glove", "a glass piece", "an unknown object"]
# T3 class_labels = ["No glass pieces or green objects", "glass piece on a tray", "Green object on the tray"]

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
        # Convert logits to probabilities (scores)
    probs = F.softmax(logits, dim=1).squeeze()  # Shape: [num_classes]

    # Zip class labels with their corresponding scores
    label_scores = list(zip(class_labels, probs.tolist()))

    # Sort by score (optional, descending)
    label_scores = sorted(label_scores, key=lambda x: x[1], reverse=True)

    # Print all label scores
    print(f"🖼️ Image: {os.path.basename(image_path)} → Predictions:")
    for label, score in label_scores:
        print(f"  {label}: {score:.4f}")
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
empty_floor_path = "W:\gasSensor_ws\others\\nav experiment obs images\\vlm test\empty floor"
obstacle_path = "W:\gasSensor_ws\others\\nav experiment obs images\\vlm test\object on the floor"

pickup_nohazard = "W:\gasSensor_ws\others\data for resnet\man_dataset\pickup task\\test\exp1 - no hazard 25 runs 16_3_25 (inc. for v3 train)"
pichup_hazard = "W:\gasSensor_ws\others\data for resnet\man_dataset\pickup task\\test\exp2 - hazard 25 runs 13_3_25 (inc. in v3 train)"

CS_hazard = "W:\gasSensor_ws\others\data for resnet\man_dataset\Dataset_chemspeed\\test\\nh_test_24_03_25"

intro = "W:\gasSensor_ws\others\Intro_test"

predict_folder(intro)
