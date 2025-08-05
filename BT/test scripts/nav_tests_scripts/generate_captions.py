import os
import csv

# ✅ Define dataset directory
dataset_dir = "W:/gasSensor_ws/GasSensor_ws/data_v2/total/"  # Change this to your dataset path
output_csv = "captions.csv"

# ✅ Open CSV file for writing
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image", "text"])  # CSV Header

    # ✅ Scan through each class folder
    for class_folder in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_folder)

        # ✅ Skip non-directory files
        if not os.path.isdir(class_path):
            continue

        # ✅ Generate caption from folder name
        caption_text = class_folder.replace("_", " ")  # Example: "rack_missing_vials" -> "rack missing vials"

        # ✅ Scan for images inside the folder
        for image_file in sorted(os.listdir(class_path)):
            if image_file.lower().endswith((".jpg", ".png", ".jpeg")):  # Modify if needed
                image_path = os.path.join(class_path, image_file)

                # ✅ Write image path and caption to CSV
                writer.writerow([image_path, caption_text])

print(f"✅ Captions CSV generated successfully: {output_csv}")
