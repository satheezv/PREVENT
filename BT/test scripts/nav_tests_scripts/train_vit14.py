import os
import json
import torch
import torch.nn.functional as F
from PIL import Image
from datasets import Dataset
from transformers import CLIPProcessor, CLIPModel, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from collections import Counter

# -------- Step 1: Load Dataset from JSON File -------- #
json_file = "W:\gasSensor_ws\GasSensor_ws\data_v2\VLM Training\dataset.json"  # Change this to your actual JSON file path

# Load JSON dataset
with open(json_file, "r") as f:
    dataset_entries = json.load(f)

# Convert to Hugging Face Dataset format
dataset = Dataset.from_list(dataset_entries)

# Print dataset label counts
label_counts = Counter(entry["text"] for entry in dataset_entries)
print("\n🔹 Dataset Label Distribution:")
for label, count in label_counts.items():
    print(f"   {label}: {count} images")

# Load CLIP processor
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

def preprocess_function(example):
    image = Image.open(example["image_path"]).convert("RGB")
    inputs = processor(text=[example["text"]], images=image, return_tensors="pt", padding=True)
    return {
        "pixel_values": inputs["pixel_values"].squeeze(0),
        "input_ids": inputs["input_ids"].squeeze(0),
    }

dataset = dataset.map(preprocess_function)

# -------- Step 2: Load CLIP Model with LoRA -------- #
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

# Apply LoRA for efficient fine-tuning
lora_config = LoraConfig(
    r=8,  # Low-rank dimension
    lora_alpha=16,  # Scaling factor
    target_modules=["self_attn.k_proj", "self_attn.q_proj", "self_attn.v_proj", "self_attn.out_proj", 
                    "mlp.fc1", "mlp.fc2"],  # Only Linear layers
    lora_dropout=0.1,
    bias="none"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -------- Step 3: Define Contrastive Loss for CLIP -------- #
class CLIPTrainer(Trainer):
    """ Custom Trainer for CLIP with contrastive loss. """
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # <-- Accept extra arguments
        outputs = model(**inputs)

        # Extract similarity logits
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        # Create labels: each text should match its corresponding image
        labels = torch.arange(logits_per_image.shape[0]).to(logits_per_image.device)

        # Compute contrastive loss using cross-entropy
        loss_image = F.cross_entropy(logits_per_image, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_image + loss_text) / 2  # Average loss

        return (loss, outputs) if return_outputs else loss



# -------- Step 4: Define Training Arguments -------- #
training_args = TrainingArguments(
    output_dir="./clip-lora-finetuned",
    eval_strategy="no",  # Disable evaluation
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=1,  # Adjust for CPU memory
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    fp16=False  # Set True if using GPU
)

# -------- Step 5: Initialize Trainer and Start Training -------- #
trainer = CLIPTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

# -------- Step 6: Save Fine-Tuned Model -------- #
model.save_pretrained("./clip-lora-finetuned")
processor.save_pretrained("./clip-lora-finetuned")

print("✅ Fine-tuning complete. Model saved to './clip-lora-finetuned'.")
