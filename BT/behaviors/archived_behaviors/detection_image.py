import py_trees
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
from .terminate_tree import TerminateTree

class DetectHazard(py_trees.behaviour.Behaviour):
    def __init__(self, name, labels, label_to_detect):
        super(DetectHazard, self).__init__(name)
        self.labels = labels
        self.label_to_detect = label_to_detect
        self.image_path = "W:\gasSensor_ws\GasSensor_ws\\rack missing vials.png"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32",token="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")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32",token="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").to(self.device)

    def check_for_hazard(self):
        # Load the image using PIL
        try:
            pil_image = Image.open(self.image_path).convert("RGB")
            print(f"Processing image: {self.image_path}")
        except Exception as e:
            print(f"Error loading image: {e}")
            return "Error", {}

        # Prepare the inputs for OWL-ViT
        inputs = self.processor(text=self.labels, images=pil_image, return_tensors="pt").to(self.device)

        # Run inference on the image
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Process the outputs
        logits = outputs.logits  # Classification logits
        boxes = outputs.pred_boxes  # Predicted bounding boxes

        # Calculate confidence scores for all labels
        scores = torch.softmax(logits[0], dim=1).cpu().numpy()  # Softmax to get probabilities
        label_scores = {self.labels[i]: scores[:, i].max().item() for i in range(len(self.labels))}

        # Print the highest score for each label
        print("Highest scores for all labels:")
        for label, score in label_scores.items():
            print(f"  {label}: {score:.2f}")

        # Determine the best match
        best_match_idx = scores.argmax()
        best_match_score = scores.flatten()[best_match_idx]
        best_match_label = self.labels[best_match_idx % len(self.labels)]

        if best_match_score > 0.7:  # Adjust the threshold as needed
            box = boxes[0][best_match_idx // len(self.labels)].cpu().numpy()
            print(f"Detected '{best_match_label}' with confidence {best_match_score:.2f} at {box}")
            return best_match_label, label_scores
        else:
            print("No hazard detected above the threshold.")
            return "No match", label_scores

    def update(self):
        best_match_label, label_scores = self.check_for_hazard()
        print("Best match label:", best_match_label)
        if best_match_label == self.label_to_detect:
            print(f"Hazard '{self.label_to_detect}' detected in the image.")
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE
