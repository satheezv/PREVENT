import torch
import torchvision.transforms as transforms
from PIL import Image
import json

class ResNetBehaviorTester:
    def __init__(self, model_path, class_map_path=None, device=None):
        """
        Load a trained ResNet model for behavior tree-based decision making.
        :param model_path: Path to the trained model file (.pth).
        :param class_map_path: Path to a JSON file mapping class indices to labels (optional).
        :param device: Device to run the model on ('cuda' or 'cpu').
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        
        # Load class mapping if available
        self.class_map = None
        if class_map_path:
            with open(class_map_path, 'r') as f:
                self.class_map = json.load(f)
        
        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image_path):
        """
        Predict the behavior class from an image.
        :param image_path: Path to the test image.
        :return: Predicted class label or index.
        """
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = predicted.item()
        
        return self.class_map.get(str(predicted_class), predicted_class) if self.class_map else predicted_class

# Example usage:
# tester = ResNetBehaviorTester(model_path='model.pth', class_map_path='class_map.json')
# behavior = tester.predict('test_image.jpg')
# print("Predicted Behavior:", behavior)
