import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# -------- Load Fine-Tuned Model -------- #
model_path = "W:\gasSensor_ws\GasSensor_ws\clip-lora-finetuned"
model = CLIPModel.from_pretrained(model_path)
processor = CLIPProcessor.from_pretrained(model_path)

def test_model(image_path, texts):
    """Loads an image, processes it with CLIP, and returns the best match."""
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1)
    best_match_idx = probs.argmax()
    return texts[best_match_idx]

# -------- Example Test -------- #
test_image = "W:\gasSensor_ws\others\_empty.png"  # Replace with an actual test image path
test_classes = ["empty floor", "obstacle on the floor"]

best_match = test_model(test_image, test_classes)
print(f"Best match: {best_match}")
