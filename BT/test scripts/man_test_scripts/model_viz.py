import torch
import torchvision.models as models

# Load model
model = models.resnet18()
model.load_state_dict(torch.load("ztests\man_test_scripts\\resnet18_rack_detection_v3.pth"))

# Convert model to TorchScript for better compatibility
scripted_model = torch.jit.script(model)
scripted_model.save("resnet18_rack_detection_v3_scripted.pth")
