import py_trees
import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import pyrealsense2 as rs
import numpy as np
import cv2
from .terminate_tree import TerminateTree

class DetectHazard(py_trees.behaviour.Behaviour):
    def __init__(self, name, labels, label_to_detect):
        super(DetectHazard, self).__init__(name)
        self.labels = labels
        self.label_to_detect = label_to_detect
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(self.device)
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline.start(config)

    def check_for_hazard(self):
        # Wait for a coherent frame from the camera
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return "No frame"

        # Convert RealSense frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Convert the frame to a PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        pil_image.save(f"{self.label_to_detect}.png")

        # Prepare the inputs for OWL-ViT
        inputs = self.processor(text=self.labels, images=pil_image, return_tensors="pt").to(self.device)

        # Run inference on the frame
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Process the outputs
        logits = outputs.logits  # Classification logits
        boxes = outputs.pred_boxes  # Predicted bounding boxes

        # Get the best match for the label of interest
        scores = torch.softmax(logits[0], dim=1)  # Confidence scores
        label_scores = {self.labels[i]: scores[:, i].max().item() for i in range(len(self.labels))}

        # Print the highest score for each label
        print("Highest scores for all labels:")
        for label, score in label_scores.items():
            print(f"  {label}: {score:.2f}")
        best_match_idx = scores[:, self.labels.index(self.label_to_detect)].argmax().item()
        best_match_score = scores[best_match_idx, self.labels.index(self.label_to_detect)].item()
        

        if best_match_score > 0.9:  # Adjust the threshold as needed
            box = boxes[0][best_match_idx].cpu().numpy()
            print(f"Detected '{self.label_to_detect}' with confidence {best_match_score:.2f} at {box}")
            return self.label_to_detect
        else:
            print("No hazard detected.")
            return "No match"

    def update(self):
        best_match_label = self.check_for_hazard()
        print("Best match label:", best_match_label)
        if best_match_label == self.label_to_detect:
            print(f"Hazard '{self.label_to_detect}' detected by RealSense camera.")
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE
