import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# ✅ Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Define dataset directory
dataset_dir = "W:/gasSensor_ws/GasSensor_ws/data_v2/unbalanced"

# ✅ Automatically detect classes from folder names
class_folders = [f for f in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, f))]
class_map = {class_name: idx for idx, class_name in enumerate(sorted(class_folders))}
print(f"Detected Classes: {class_map}")

# ✅ Compute Class Weights for Balanced Loss
class_counts = {class_name: len(glob.glob(os.path.join(dataset_dir, class_name, '*.jpg'))) for class_name in class_map}
class_weights = torch.tensor([1.0 / class_counts[class_name] for class_name in class_map], dtype=torch.float32)
class_weights = class_weights.to(device)

# ✅ Optimized Data Augmentation for EfficientNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(20),  # Reduced rotation to avoid extreme distortions
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Smaller translation range
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
])

# ✅ Custom Dataset
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        for class_name, label in class_map.items():
            class_path = os.path.join(root_dir, class_name)
            image_paths = sorted(glob.glob(os.path.join(class_path, "*.jpg")))
            for img_path in image_paths:
                self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

# ✅ Create dataset & dataloaders
dataset = ImageFolderDataset(dataset_dir, transform=transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # Increased batch size
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
print(f"Training samples: {train_size}, Validation samples: {val_size}")

# ✅ Load Pretrained EfficientNet Model
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_map))
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)  # Reduced learning rate
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# ✅ Training Function with Gradient Clipping and Early Stopping
def train_efficientnet(model, train_loader, val_loader, epochs=50, patience=10):
    best_val_acc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
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

        # ✅ Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "best_efficientnet.pth")
            print("✅ Model saved with highest validation accuracy!")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            # break

        scheduler.step(total_loss)
    torch.save(model.state_dict(), "best_efficientnet_final.pth")
    print("✅ Final Model saved!")

# ✅ Train Model
train_efficientnet(model, train_loader, val_loader, epochs=50, patience=10)

# ✅ Load Best Model for Evaluation
model.load_state_dict(torch.load("best_efficientnet_final.pth"))
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
