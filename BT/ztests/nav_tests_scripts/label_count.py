from collections import Counter
import json

json_file = "W:\gasSensor_ws\GasSensor_ws\data_v2\VLM Training\dataset.json"
with open(json_file, "r") as f:
    dataset_entries = json.load(f)

label_counts = Counter(entry["text"] for entry in dataset_entries)
print(label_counts)