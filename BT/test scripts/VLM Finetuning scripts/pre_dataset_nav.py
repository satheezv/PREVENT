import os
import csv

# Folder containing the images
folder_name = "dataset/navigation"
# base_dir = "./"  # Adjust if your folder lives elsewhere
folder_path = os.path.join(folder_name)

# Output CSV file
output_csv = "navigation_dataset.csv"

with open(output_csv, mode="w", newline="", encoding="utf-8") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["filepath", "caption"])

    if not os.path.isdir(folder_path):
        print(f"Warning: Folder not found - {folder_path}")
    else:
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                full_path = os.path.join(folder_path, filename).replace("\\", "/")

                name_without_ext = os.path.splitext(filename)[0]  # remove file extension

                # Remove trailing _number
                if "_" in name_without_ext:
                    base_text = name_without_ext.rsplit("_", 1)[0]
                else:
                    base_text = name_without_ext

                # Remove trailing 'on the floor' (if present)
                cleaned_text = base_text.replace(" on the floor", "")

                # Caption = folder_name + cleaned_text
                caption = f"{folder_name} {cleaned_text}"
                writer.writerow([full_path, caption])

print(f"CSV file saved to: {output_csv}")