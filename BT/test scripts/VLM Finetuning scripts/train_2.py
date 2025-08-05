import os
import pandas as pd
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader, Dataset, random_split
from torch import nn, optim
from tqdm import tqdm
import torch
import wandb
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
# Initialize wandb
wandb.init(project="clip-finetuning", name="clip-vit-large-patch14")


train_df = pd.read_csv("split_train.csv")
eval_df = pd.read_csv("split_val.csv")
test_df = pd.read_csv("split_test.csv")

train_df = train_df.sample(1000)
eval_df = eval_df.sample(200)

# Dataset class
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
        return image, caption

def clip_collate_fn(batch):
    images, captions = zip(*batch)
    return list(images), list(captions)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model.eval()
for param in model.parameters():
    param.requires_grad = True 

# Prepare datasets and loaders
train_dataset = CLIPImageTextDataset(train_df)
eval_dataset = CLIPImageTextDataset(eval_df)
test_dataset = CLIPImageTextDataset(test_df)

num_workers=4 

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=clip_collate_fn)
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, collate_fn=clip_collate_fn)

# Build class weights
unique_labels = train_df['caption'].unique()
label_to_index = {label: i for i, label in enumerate(unique_labels)}
train_labels = train_df['caption'].map(label_to_index).values

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(train_labels), y=train_labels)
class_weight_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
loss_fn = nn.CrossEntropyLoss(weight=class_weight_tensor)
optimizer = optim.Adam(model.parameters(), lr=5e-6,betas=(0.9,0.98),eps=1e-6,weight_decay=0.001)
# loss_fn = nn.CrossEntropyLoss()

EPOCHS = 50
best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.eval()
    total_loss = 0.0

    for i, (images, captions) in enumerate(tqdm(train_loader, desc=f"[Train] Epoch {epoch + 1}")):
        text_inputs = processor(text=list(unique_labels), return_tensors="pt", padding=True, truncation=True).to(device)
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

        # Process and extract image features (only images)
        image_inputs = processor(images=images, return_tensors="pt").to(device)
        image_features = model.get_image_features(**image_inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        # Compute logits and loss
        logits = image_features @ text_features.T
        ground_truth = torch.tensor([label_to_index[cap] for cap in captions], dtype=torch.long).to(device)
        loss = loss_fn(logits, ground_truth)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        global_step = epoch * len(train_loader) + i
        wandb.log({"train/loss": loss.item(), "step": global_step})

    avg_train_loss = total_loss / len(train_loader)
    print(f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f}")

    # Validation
    # model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for j, (images, captions) in enumerate(tqdm(eval_loader, desc=f"[Val] Epoch {epoch + 1}")):
            text_inputs = processor(text=list(unique_labels), return_tensors="pt", padding=True, truncation=True).to(device)
            text_features = model.get_text_features(**text_inputs) 

            image_inputs = processor(images=images, return_tensors="pt").to(device)
            image_features = model.get_image_features(**image_inputs)

            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

            logits = image_features @ text_features.T
            ground_truth = torch.tensor([label_to_index[cap] for cap in captions], dtype=torch.long).to(device)
            loss = loss_fn(logits, ground_truth)
            val_loss += loss.item()
            val_step = epoch * len(eval_loader) + j
            wandb.log({"val/loss": loss.item(), "step": val_step})

    avg_val_loss = val_loss / len(eval_loader)
    print(f"[Epoch {epoch + 1}] Validation Loss: {avg_val_loss:.4f}")

    # Log to wandb
    wandb.log({"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        model.save_pretrained("clip_finetuned_best")
        processor.save_pretrained("clip_finetuned_best")
        print("Saved best model.")

# Save final model
model.save_pretrained("clip_finetuned_final")
processor.save_pretrained("clip_finetuned_final")
print("Training complete.")
