from PIL import Image
import os

def verify_images(image_dir):
    corrupted_files = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            if file.endswith(('png', 'jpg', 'jpeg')):
                try:
                    img = Image.open(os.path.join(root, file))
                    img.verify()  # Verifies without loading entire image
                except (OSError, IOError):
                    corrupted_files.append(os.path.join(root, file))

    if corrupted_files:
        print(f"Corrupted files found: {len(corrupted_files)}")
        for file in corrupted_files:
            print(f" - {file}")
    else:
        print("No corrupted files found.")

# Example usage
verify_images('W:\gasSensor_ws\others\data for resnet\man_dataset\Dataset_chemspeed\\train\\hazard')
