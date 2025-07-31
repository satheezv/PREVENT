import os
import shutil

# Update these paths
source_folder = "W:\gasSensor_ws\others\data v3\extra train"      # Folder where images are copied from
destination_folder = "W:\gasSensor_ws\others\data v3\empty_floor_train"  # Folder where images are copied to

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Valid image file extensions
valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")

# Get existing images in destination folder
existing_files = [f for f in os.listdir(destination_folder) if f.lower().endswith(valid_extensions)]

# Find the highest numbered file in the destination folder
existing_numbers = []
for file in existing_files:
    try:
        num_part = int(file.split("_")[-1].split(".")[0])  # Extracts the numeric part
        existing_numbers.append(num_part)
    except ValueError:
        continue  # Skip files that don't match the pattern

# Determine the next available image number
next_number = max(existing_numbers, default=0) + 1

# Get images from source folder
source_images = [f for f in os.listdir(source_folder) if f.lower().endswith(valid_extensions)]

# Copy and rename images to the destination folder
for image in source_images:
    new_filename = f"image_{next_number:04d}.jpg"  # Format as image_0001.jpg
    source_path = os.path.join(source_folder, image)
    destination_path = os.path.join(destination_folder, new_filename)

    shutil.copy2(source_path, destination_path)  # Copy with metadata
    print(f"✅ Copied: {image} → {new_filename}")

    next_number += 1  # Increment for next image

print("🎉 All images copied and renamed successfully!")
