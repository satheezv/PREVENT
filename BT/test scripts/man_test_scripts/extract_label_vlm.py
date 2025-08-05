

import os
import json
import re

# Define the folder containing images
folder_path = "W:\gasSensor_ws\others\data for resnet\\nav_dataset\combined"   # Change this to your actual folder

# Define the new image path for JSON output
image_base_path = "/content/drive/MyDrive/nav_dataset/combined"

# Get all image files
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

# Function to remove index numbers and extract class names
def clean_class_name(filename):
    # Remove file extension
    name_without_ext = os.path.splitext(filename)[0]
    
    # Remove trailing numbers using regex (e.g., "object_23" → "object")
    clean_name = re.sub(r'_\d+$', '', name_without_ext)
    
    return clean_name

# Create dataset dictionary
dataset_dict = []

for image in image_files:
    class_name = clean_class_name(image)  # Extract class name
    image_path = os.path.join(image_base_path, image)  # Set correct image path

    dataset_dict.append({"image": image_path, "text": class_name})

# Save to JSON file
json_path = os.path.join(folder_path, "dataset.json")
with open(json_path, "w", encoding="utf-8") as json_file:
    json.dump(dataset_dict, json_file, indent=4)

print(f"✅ Dataset JSON file created: {json_path}")
