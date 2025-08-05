import py_trees
import torch
import clip
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
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.text_inputs = torch.cat([clip.tokenize(f"a photo of a {label}") for label in labels]).to(self.device)
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
        
        # Convert the frame to a PIL Image and preprocess it
        pil_image = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        pil_image.save(f"{self.label_to_detect}.png")
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        
        # Run inference on the frame
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(self.text_inputs)
            similarities = (image_features @ text_features.T).softmax(dim=-1).cpu().numpy()
        
        # Get the best matching label

        print("values", similarities)
        best_match_idx = similarities.argmax()
        best_match_label = self.labels[best_match_idx]
        
        print(f"Best match: {best_match_label}")
        return best_match_label
    
    def update(self):
        best_match_label = self.check_for_hazard()
        print ("best_match_label: ", best_match_label)
        if best_match_label == self.label_to_detect:
            print("Hazard detected by RealSense camera.")
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE