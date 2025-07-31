import torch
import clip
from PIL import Image
import pyrealsense2 as rs
import numpy as np
import cv2
import pythoncom


pythoncom.CoInitialize()
# Load the CLIP model and set device
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Define the text prompts
labels = ["vial on the floor", "liquid spillage", "solid powder spillage", 
          "sharp needle", "broken glass", "floor", "scissor"]
text_inputs = torch.cat([clip.tokenize(f"a photo of a {label}") for label in labels]).to(device)

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

        # Convert the frame to a PIL Image and preprocess it
        pil_image = Image.fromarray(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
        image_input = preprocess(pil_image).unsqueeze(0).to(device)

        # Run inference on the frame
        with torch.no_grad():
            image_features = model.encode_image(image_input)
            text_features = model.encode_text(text_inputs)
            similarities = (image_features @ text_features.T).softmax(dim=-1).cpu().numpy()

        # Get the best matching label
        best_match_idx = similarities.argmax()
        best_match_label = labels[best_match_idx]

        # Display the label on the frame
        cv2.putText(color_image, f"Best match: {best_match_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame with the detected label
        cv2.imshow("Real-Time CLIP Detection with RealSense", color_image)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the RealSense pipeline and close windows
    pipeline.stop()
    cv2.destroyAllWindows()
    pythoncom.CoUninitialize()

