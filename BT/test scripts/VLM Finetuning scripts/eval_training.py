import os
import pandas as pd
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
import json

# Load the fine-tuned model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("clip_finetuned_final").to(device)
processor = CLIPProcessor.from_pretrained("clip_finetuned_final")

# Load the test dataset CSV
test_df = pd.read_csv("real_testing_dataset.csv")

# Define Dataset class
class CLIPImageTextDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        caption = row["caption"]  # Ground truth label
        image = Image.open(img_path).convert("RGB")
        return image, caption, img_path

# Collate function for processor
def clip_collate_fn(batch):
    images, captions, paths = zip(*batch)
    return list(images), list(captions), list(paths)

# Candidate labels for prediction
candidate_labels = [
    "chemspeed clear", "chemspeed hazard",
    "capping clear", "capping spillage", "capping obstruction", "capping missing cap", "capping missing vial",
    "navigation object glove", "navigation empty floor", "navigation other objects", "navigation object tool", "navigation object vial"
]

# Setup test dataloader
test_dataset = CLIPImageTextDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=clip_collate_fn)

# Evaluation and result collection
results = []
correct = 0
total = 0

print("\n Evaluating on test set...")
model.eval()

with torch.no_grad():
    for images, captions, paths in tqdm(test_loader, desc="Testing"):
        inputs = processor(images=images, text=candidate_labels, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=1)

        pred_idx = probs.argmax(dim=1).item()
        pred_label = candidate_labels[pred_idx]
        pred_conf = probs[0][pred_idx].item()

        ground_truth = captions[0]
        is_correct = pred_label.strip().lower() == ground_truth.strip().lower()

        results.append({
            "image": paths[0],
            "ground_truth": ground_truth,
            "predicted_label": pred_label,
            "confidence": round(pred_conf, 4),
            "correct": is_correct
        })

        if is_correct:
            correct += 1
        total += 1

# Save results to a JSON file
with open("test_predictions.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# Print overall accuracy
accuracy = correct / total if total > 0 else 0.0
print(f"\n Evaluation complete. Accuracy: {accuracy:.2%} ({correct}/{total})")
