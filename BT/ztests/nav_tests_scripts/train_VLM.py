import os
import torch
import clip
import pandas as pd
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Dataset class
class HazardDataset(Dataset):
    def __init__(self, captions_file, transform=None):
        self.captions = pd.read_csv(captions_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_path = self.captions.iloc[idx]['image']
        text = self.captions.iloc[idx]['text']
        
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        
        # Tokenize text
        text = clip.tokenize(text)[0]
        
        return image, text

# Set paths
CAPTIONS_FILE = "W:\gasSensor_ws\GasSensor_ws\data_v2\\total\captions.csv"

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275)),  # CLIP normalization
])

# Load dataset
dataset = HazardDataset(CAPTIONS_FILE, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Optimizer & Loss Function
optimizer = optim.Adam(model.parameters(), lr=5e-6)
loss_fn = nn.CrossEntropyLoss()

# Fine-tuning loop
EPOCHS = 25
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for images, texts in tqdm(dataloader):
        images = images.to(device)
        texts = texts.to(device)

        # Encode images and texts
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)

        # Compute similarity
        logits = (image_features @ text_features.T)

        # Compute loss
        labels = torch.arange(len(logits)).to(device)
        loss = loss_fn(logits, labels)
        total_loss += loss.item()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader)}")

# Save fine-tuned model
torch.save(model.state_dict(), "fine_tuned_clip.pth")
