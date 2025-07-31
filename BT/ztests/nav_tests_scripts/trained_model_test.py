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
import py_trees

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.load_state_dict(torch.load("fine_tuned_clip.pth", map_location=device))
model.eval()

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

# Testing function
def test_clip_model(image_path, text_descriptions):
    image = Image.open(image_path).convert("RGB")
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_inputs = torch.cat([clip.tokenize(desc) for desc in text_descriptions]).to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)
        similarities = (image_features @ text_features.T).softmax(dim=-1).cpu().numpy()
    
    best_match_idx = similarities.argmax()
    best_match_text = text_descriptions[best_match_idx]
    print(f"Best match: {best_match_text}")
    return best_match_text

# DetectHazard class
class DetectHazard(py_trees.behaviour.Behaviour):
    def __init__(self, name, labels, label_to_detect):
        super(DetectHazard, self).__init__(name)
        self.labels = labels
        self.label_to_detect = label_to_detect
        self.text_inputs = torch.cat([clip.tokenize(f"a photo of {label}") for label in labels]).to(device)
    
    def check_for_hazard(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image_input = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(self.text_inputs)
            similarities = (image_features @ text_features.T).softmax(dim=-1).cpu().numpy()
        
        best_match_idx = similarities.argmax()
        best_match_label = self.labels[best_match_idx]
        print(f"Best match: {best_match_label}")
        return best_match_label
    
    def update(self, image_path):
        best_match_label = self.check_for_hazard(image_path)
        print("Best Match Label:", best_match_label)
        if best_match_label == self.label_to_detect:
            print("Hazard detected.")
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE

# Example test
# CAPTIONS_FILE = "W:\gasSensor_ws\GasSensor_ws\data\Modified_Captions_CSV_1.csv"
test_image_path = "W:\gasSensor_ws\GasSensor_ws\data\perfect_rack\image_0075.jpg"
text_descriptions = ["perfect_rack", "rack_missing_vials", "vial_missing_caps"]

detector = DetectHazard(name="Hazard Detector", labels=text_descriptions, label_to_detect="rack missing one vial")
result = detector.update(test_image_path)
print("Detection Result:", result)
