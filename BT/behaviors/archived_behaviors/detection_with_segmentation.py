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
    
    def process_vials(self, color_image):
        """
        Detect and classify each vial in the image.
        """
        vial_results = []
        
        # Convert to grayscale and find contours
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        number_of_vials = len(contours)
        print(f"Number of vials detected: {number_of_vials}")
        
        for contour in contours:
            # Extract bounding box for each vial
            x, y, w, h = cv2.boundingRect(contour)
            cropped_vial = color_image[y:y+h, x:x+w]
            
            # Convert cropped vial to PIL Image and preprocess
            pil_image = Image.fromarray(cv2.cvtColor(cropped_vial, cv2.COLOR_BGR2RGB))
            image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            
            # Run inference with CLIP
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(self.text_inputs)
                similarities = (image_features @ text_features.T).softmax(dim=-1).cpu().numpy()
            
            # Get the best matching label
            best_match_idx = similarities.argmax()
            best_match_label = self.labels[best_match_idx]
            vial_results.append(best_match_label)
        
        return vial_results
    
    def check_for_hazard(self):
        """
        Capture a frame from the RealSense camera and check for hazards.
        """
        # Wait for a coherent frame from the camera
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            return "No frame"
        
        # Convert RealSense frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())
        
        # Process vials in the captured frame
        vial_results = self.process_vials(color_image)
        
        # Check if any vial is uncapped
        for idx, result in enumerate(vial_results):
            print(f"Vial {idx + 1}: {result}")
            if result == self.label_to_detect:
                print("Hazard detected: Vial without a cap.")
                return self.label_to_detect
        
        print("No hazards detected.")
        return None
    
    def update(self):
        best_match_label = self.check_for_hazard()
        print("Best Match Label:", best_match_label)
        if best_match_label == self.label_to_detect:
            print("Hazard detected by RealSense camera.")
            return py_trees.common.Status.SUCCESS
        else:
            return py_trees.common.Status.FAILURE
