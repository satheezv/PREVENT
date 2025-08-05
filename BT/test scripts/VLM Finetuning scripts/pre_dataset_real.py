import os
import csv

# Define the base directory
base_dir = "dataset/real_testing"
output_csv = "real_testing_dataset.csv"

# Open CSV file for writing
with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["filepath", "caption"])  # CSV header

    # Traverse all subfolders
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Process all image files in each subfolder
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                full_path = os.path.join(folder_path, filename).replace("\\", "/")
                writer.writerow([full_path, folder_name.replace("_", " ")])  # Optional: prettify caption

print(f"✅ CSV saved to {output_csv}")
