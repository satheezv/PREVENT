import os
import pandas as pd
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torch
import json

# Load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("clip_finetuned_final").to(device)
processor = CLIPProcessor.from_pretrained("clip_finetuned_final")

# Load dataset
test_df = pd.read_csv("real_testing_dataset.csv")

class CLIPImageTextDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["filepath"]
        caption = row["caption"]
        image = Image.open(img_path).convert("RGB")
        return image, caption, img_path

def clip_collate_fn(batch):
    images, captions, paths = zip(*batch)
    return list(images), list(captions), list(paths)

candidate_labels = [
    "chemspeed clear", "chemspeed hazard",
    "capping clear", "capping spillage", "capping obstruction", "capping missing cap", "capping missing vial",
    "navigation object glove", "navigation empty floor", "navigation other objects", "navigation object tool", "navigation object vial"
]

# Logical class grouping for flexible correctness
def is_prediction_correct(gt, pred):
    gt, pred = gt.lower().strip(), pred.lower().strip()

    if gt.startswith("chemspeed"):
        return pred.startswith("chemspeed") and not (gt == "chemspeed hazard" and pred == "chemspeed clear")

    if gt.startswith("capping"):
        if gt == "capping clear":
            return pred == "capping clear"
        else:
            return pred != "capping clear"

    if gt.startswith("navigation"):
        if gt == "navigation empty floor":
            return pred == "navigation empty floor"
        else:
            return pred != "navigation empty floor"

    return False  # default fallback

# DataLoader
test_dataset = CLIPImageTextDataset(test_df)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=clip_collate_fn)

# Evaluation
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
        is_correct = is_prediction_correct(ground_truth, pred_label)

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

# Save results
with open("test_predictions.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

# Accuracy
accuracy = correct / total if total > 0 else 0.0
print(f"\n ✅ Evaluation complete. Accuracy: {accuracy:.2%} ({correct}/{total})")
