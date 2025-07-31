import os
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

def test_resnet18(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)  # Two classes: Empty Floor vs Obstruction
    # model.load_state_dict(torch.load(r"W:\\gasSensor_ws\\GasSensor_ws\src\BT\\ztests\\man_test_scripts\\resnet18_rack_detection_v2.pth", map_location=device))
    model.load_state_dict(torch.load("W:\gasSensor_ws\GasSensor_ws\src\BT\ztests\man_test_scripts\\resnet18_rack_detection_v3.pth", map_location=device))

    model.eval().to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
        class_names = ['hazard', 'no hazard']

    print(f"🔍 ResNet-18 Prediction for {image_path}: {class_names[predicted.item()]}")

    return class_names[predicted.item()]

def predict_folder_images(folder_path):
    # Supported image formats
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    hazard = 0
    no_hazard = 0

    # Iterate through all images in the folder
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in valid_extensions):
            image_path = os.path.join(folder_path, filename)
            prediction = test_resnet18(image_path)
            if prediction == "hazard":
                hazard += 1
            else:
                no_hazard += 1
            print(f"{filename} -> {prediction}")
    
    print(f"Total hazard images: {hazard}")
    print(f"Total no hazard images: {no_hazard}")
        



# Example usage
if __name__ == "__main__":
    folder_path = r"W:\\gasSensor_ws\\others\data for resnet\\man_dataset\\test"  # Update this path to your image folder
    predict_folder_images(folder_path)
