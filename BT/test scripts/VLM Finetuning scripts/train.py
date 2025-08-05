import pandas as pd
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

class CLIPImageTextDataset(Dataset):
    def __init__(self, csv_path):
        import pandas as pd
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Replace 'image_path' with the actual column name
        img_path = row["filepath"]  # <- use the correct column name
        caption = row["caption"]
        image = Image.open(img_path).convert("RGB")
        return image, caption

def clip_collate_fn(batch):
    images, captions = zip(*batch)
    return list(images), list(captions)  # return as lists for processor to handle

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

dataset = CLIPImageTextDataset("chemspeed_dataset.csv")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=clip_collate_fn)

optimizer = optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = nn.CrossEntropyLoss()

EPOCHS = 3

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, captions in tqdm(dataloader, desc=f"Epoch {epoch + 1}"):
        # Process images and texts together
        inputs = processor(images=images, text=captions, return_tensors="pt", padding=True, truncation=True).to(device)

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

        loss = (loss_fn(logits_per_image, ground_truth) + loss_fn(logits_per_text, ground_truth)) / 2
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()

    print(f"[Epoch {epoch + 1}] Average Loss: {total_loss / len(dataloader):.4f}")



model.save_pretrained("clip_finetuned")
processor.save_pretrained("clip_finetuned")