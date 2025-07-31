import py_trees
from behaviors.sensor.GetVisionPrediction import CLIPImageTextMatching
from behaviors.sensor.CaptureImage import CaptureImageBehavior  # Import CaptureImageBehavior
from huggingface_hub import login

# 🔹 Authenticate with Hugging Face
HF_TOKEN = "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"  # Replace with your token
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("❌ Hugging Face token not found. Please set HF_TOKEN.")

# 🔹 Define object labels for CLIP matching
text_labels = ["a vial", "a glove", "a screwdriver", "a tissue paper", "an unknown object"]

# 🔹 Create behavior nodes
capture_behavior = CaptureImageBehavior(name="Capture Image")  # Capture image from RealSense
clip_behavior = CLIPImageTextMatching(name="CLIP Matcher", text_labels=text_labels)

# 🔹 Build a behavior tree
root = py_trees.composites.Sequence("Root", memory=True)
root.add_child(capture_behavior)  # Step 1: Capture an image
root.add_child(clip_behavior)  # Step 2: Use CLIP to recognize objects in the captured image

# 🔹 Run the behavior tree
tree = py_trees.trees.BehaviourTree(root)
print("\n🔹 Running Behavior Tree...\n")
tree.tick_tock(500)  # Tick every 500ms
