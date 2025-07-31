import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image
from torchvision.models.video import r3d_18

# ✅ Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Define dataset directory
dataset_dir = "W:/gasSensor_ws/GasSensor_ws/data_v2/core"

# ✅ Automatically detect classes from folder names
class_folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
class_map = {class_name: idx for idx, class_name in enumerate(sorted(class_folders))}
print(f"Detected Classes: {class_map}")

# ✅ Compute Class Weights for Balanced Loss
class_counts = {class_name: len(glob.glob(os.path.join(dataset_dir, class_name, '*.jpg'))) for class_name in class_map}
class_weights = torch.tensor([1.0 / class_counts[class_name] for class_name in class_map], dtype=torch.float32)
class_weights = class_weights.to(device)

# ✅ Data Augmentation for Robustness
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])

# ✅ Custom Dataset
class ImageFolderSequenceDataset(Dataset):
    def __init__(self, root_dir, transform=None, sequence_length=16):
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        self.data = []

        for class_name, label in class_map.items():
            class_path = os.path.join(root_dir, class_name)
            image_paths = sorted(glob.glob(os.path.join(class_path, "*.jpg")))

            for i in range(0, len(image_paths) - sequence_length + 1, sequence_length):
                self.data.append((image_paths[i:i+sequence_length], label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_paths, label = self.data[idx]
        images = []

        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)

        images = torch.stack(images, dim=1)  # Shape: (C, T, H, W)
        return images, torch.tensor(label, dtype=torch.long)

# ✅ Create dataset & dataloaders
dataset = ImageFolderSequenceDataset(dataset_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
print(f"Training samples: {train_size}, Validation samples: {val_size}")

# ✅ Load Pretrained ResNet3D Model
model = r3d_18(weights="KINETICS400_V1")
model.fc = nn.Linear(model.fc.in_features, len(class_map))
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ✅ Training Function
def train_resnet3d(model, train_loader, val_loader, epochs=50):
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Training Accuracy: {train_acc:.2f}%")
        
        # ✅ Validation Loop
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        val_acc = 100 * correct / total
        print(f"Validation Accuracy: {val_acc:.2f}%")

        # ✅ Save Best Model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_resnet3d.pth")
            print("✅ Model saved with highest validation accuracy!")

        scheduler.step()

# ✅ Train Model
train_resnet3d(model, train_loader, val_loader, epochs=35)

# ✅ Load Best Model for Evaluation
model.load_state_dict(torch.load("best_resnet3d.pth"))
model.eval()

# ✅ Generate Confusion Matrix
true_labels, predicted_labels = [], []
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

conf_matrix = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_map.keys(), yticklabels=class_map.keys())
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
