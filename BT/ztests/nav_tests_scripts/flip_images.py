import os
import glob
from PIL import Image

# ✅ Define dataset directory & output directories
dataset_dir = "W:/gasSensor_ws/GasSensor_ws/data_v2/core"  # Change this to your dataset path
output_x_dir = "W:/gasSensor_ws/GasSensor_ws/data_v2/flipped_x/"
output_y_dir = "W:/gasSensor_ws/GasSensor_ws/data_v2/flipped_y/"

# ✅ Ensure output directories exist
os.makedirs(output_x_dir, exist_ok=True)
os.makedirs(output_y_dir, exist_ok=True)

# ✅ Process each class folder
for class_folder in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_folder)

    # Skip non-folder entries
    if not os.path.isdir(class_path):
        continue

    # Create class-specific output directories
    os.makedirs(os.path.join(output_x_dir, class_folder), exist_ok=True)
    os.makedirs(os.path.join(output_y_dir, class_folder), exist_ok=True)

    # ✅ Read all images from the class folder
    image_paths = glob.glob(os.path.join(class_path, "*.jpg"))  # Modify extension if needed

    for img_path in image_paths:
        img = Image.open(img_path)
        
        # Extract image name and extension
        filename, ext = os.path.splitext(os.path.basename(img_path))

        # ✅ Mirror along X-axis (horizontal flip) and rename
        flipped_x = img.transpose(Image.FLIP_LEFT_RIGHT)
        save_x_path = os.path.join(output_x_dir, class_folder, f"{filename}_fx{ext}")
        flipped_x.save(save_x_path)

        # ✅ Mirror along Y-axis (vertical flip) and rename
        flipped_y = img.transpose(Image.FLIP_TOP_BOTTOM)
        save_y_path = os.path.join(output_y_dir, class_folder, f"{filename}_fy{ext}")
        flipped_y.save(save_y_path)

print("✅ Mirroring completed! Flipped images saved with '_fx' and '_fy' suffixes.")
