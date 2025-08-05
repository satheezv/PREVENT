import os

# Define the folder path where images are stored
folder_path = "W:\gasSensor_ws\others\data for resnet\man_dataset\Dataset_chemspeed\\train\\no_hazard"  # Change this to your folder

# Define the new base name (e.g., "floor_image")
new_name = "image"

# Get all image files in the folder (jpg, png, etc.)
image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

# Rename images sequentially
for index, filename in enumerate(image_files, start=1):
    # Get the file extension
    file_extension = os.path.splitext(filename)[1]
    
    # Construct new filename
    new_filename = f"{new_name}_{index}{file_extension}"
    
    # Define old and new file paths
    old_file = os.path.join(folder_path, filename)
    new_file = os.path.join(folder_path, new_filename)

    # Rename file
    os.rename(old_file, new_file)
    print(f"Renamed: {filename} → {new_filename}")

print("✅ Renaming complete!")
