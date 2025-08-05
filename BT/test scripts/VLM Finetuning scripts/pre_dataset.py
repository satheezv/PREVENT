import os
import csv

# Define your base dataset directory
base_dir = "./dataset"
folders = ["chemspeed_clear", "chemspeed_hazard"]

# Output CSV file path
output_csv = "chemspeed_dataset.csv"

# Open CSV file for writing
with open(output_csv, mode="w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    # Write CSV header
    writer.writerow(["filepath", "caption"])

    for folder in folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            print(f"Warning: Folder not found - {folder_path}")
            continue

        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                full_path = os.path.join(folder_path, filename)
                caption = folder.replace("_", " ")  # e.g., "chemspeed_clear" -> "chemspeed clear"
                writer.writerow([full_path, caption])

print(f"CSV file saved to: {output_csv}")
