import os
import csv
import re

# Define your base dataset directory
base_dir = "./dataset"
folders = ["capping_clear", "capping_hazard"]

# Output CSV file path
output_csv = "capping_dataset.csv"

with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["filepath", "caption"])

    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"Warning: Folder not found - {folder_path}")
            continue

        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                full_path = os.path.join(folder_path, filename).replace("\\", "/")

                if folder == "capping_clear":
                    caption = "capping clear"
                else:
                    name_without_ext = os.path.splitext(filename)[0]
                    
                    # Remove trailing underscore + number (e.g., _57)
                    hazard_type = re.sub(r"_\d+$", "", name_without_ext)
                    
                    # Replace underscores with spaces
                    hazard_caption = hazard_type.replace("_", " ")
                    caption = f"capping {hazard_caption}"

                writer.writerow([full_path, caption])

print(f"CSV file saved to: {output_csv}")
