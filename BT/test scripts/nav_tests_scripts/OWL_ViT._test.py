import torch
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import pyrealsense2 as rs
import numpy as np
import cv2
import pythoncom

pythoncom.CoInitialize()

# Load the OWL-ViT model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

# Define the text prompts
labels = ["vial on the floor", "liquid spillage", "solid powder spillage", 
          "sharp needle", "broken glass", "floor", "scissor"]

# Initialize the RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Configure color stream

# Start the RealSense pipeline
pipeline.start(config)

try:
    while True:
        # Wait for a coherent frame from the camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert RealSense frame to numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Convert the frame to a PIL Image
        pil_image = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))

        # Process the image and labels
        inputs = processor(text=labels, images=pil_image, return_tensors="pt").to(device)

        # Run inference on the frame
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract predictions
        logits = outputs.logits
        boxes = outputs.pred_boxes
        
        # Find the best match
        scores = logits.softmax(dim=-1)[0, :, :-1].max(dim=-1).values
        print("Scores: ", scores)
        best_match_idx = scores.argmax().item()

        if best_match_idx < len(labels):
            best_match_label = labels[best_match_idx]
            # Display the label on the frame
            cv2.putText(color_image, f"Best match: {best_match_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            print("No valid match found.")

        # Show the frame with the detected label
        cv2.imshow("Real-Time OWL-ViT Detection with RealSense", color_image)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the RealSense pipeline and close windows
    pipeline.stop()
    cv2.destroyAllWindows()
    pythoncom.CoUninitialize()
