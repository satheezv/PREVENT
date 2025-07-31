import py_trees
import torch
import torchvision.transforms as transforms
from PIL import Image
import time
import numpy as np
import cv2
from collections import Counter
import torch
import torchvision.models as models

from utils.realSenseCamera import RealSenseCamera

class ResNetPredictionBehavior(py_trees.behaviour.Behaviour):
    """
    A PyTrees behavior that performs image classification using a trained ResNet model.
    """
    def __init__(self, name="ResNetBehaviorNav", model_path="", class_map= None, camera:RealSenseCamera=None):
        super(ResNetPredictionBehavior, self).__init__(name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 🔹 Load ResNet-18 architecture **without pretrained weights**
        self.model = models.resnet18(weights=None)  

        # 🔹 Modify final layer to match your 2-class model
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 2)  # 2 classes
        # 🔹 Load trained weights correctly
        state_dict = torch.load(model_path, map_location=self.device)

        self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()
        if class_map is None:
            self.class_map = self.load_class_map()
        else:
            self.class_map = class_map  # 🔹 Properly initialize class mapping
        
        self.camera = camera
        # Blackboard for sharing data
        self._blackboard = py_trees.blackboard.Client()
        self._blackboard.register_key(key="ResNet18_best_match", access=py_trees.common.Access.WRITE)
        self._blackboard.register_key(key="ResNet18_scores", access=py_trees.common.Access.WRITE)
        self._blackboard.register_key(key="is_obstacle_detected", access=py_trees.common.Access.WRITE)

        # Image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_class_map(self):    

        if self.name == "Resnet Binary Prediction":
            return {
                "0": "empty_floor",
                "1": "object_on_floor"
            }
        elif self.name == "ResNetBehaviorSafeGrasp":
            return {
                "0": "hazard",
                "1": "no hazard"
            }

    def initialise(self):
        """Start the camera before capturing."""
        if self.camera and not self.camera.is_running:
            self.camera.start_camera()
        self._blackboard.is_obstacle_detected = False

    def update(self):
        """Perform image classification and return the prediction."""
        try:
            if not self.camera or not self.camera.is_running:
                print(f"[{self.name}] ❌ Camera not available!")
                return py_trees.common.Status.FAILURE

            images = self.camera.capture_images(duration=1)
            if not images:
                print(f"[{self.name}] ❌ No valid image captured!")
                return py_trees.common.Status.FAILURE

            predictions = []
            confidences = []

            for img in images:
                image = Image.fromarray(img).convert('RGB')
                image = self.transform(image).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(image)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    predictions.append(predicted.item())
                    confidences.append(confidence.item())

            # Get most frequent prediction
            best_prediction, freq = Counter(predictions).most_common(1)[0]

            # Get confidence of most frequent prediction
            best_confidence = max(confidences)
            

            label = str(best_prediction)  # 🔹 Ensure label is a string
            label_text = self.class_map.get(label, "Unknown")  # 🔹 Map it to class name
            
            # Store results in blackboard
            self._blackboard.ResNet18_best_match = label_text
            self._blackboard.ResNet18_scores = best_confidence

            print(f"[{self.name}] 🔍 Prediction: {label_text} (Confidence: {best_confidence:.2f})")

            # 🔹 Fix condition to check correctly
            if label_text == "object_on_floor" or label_text == "hazard":
                print(f"[{self.name}] ⚠️ Obstacle detected!")
                self._blackboard.is_obstacle_detected = True
                
                timestamp = time.strftime("%H%M%S")
                filename = f"resnet_{timestamp}.png"
                cv2.imwrite(filename, images[0])
                print(f"Image saved as {filename}")
                return py_trees.common.Status.SUCCESS
            else:
                print(f"[{self.name}] ✅ No obstacles detected.")
                return py_trees.common.Status.RUNNING
        
        except Exception as e:
            print(f"[{self.name}] ❌ Error: {e}")
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        """Cleanup if needed."""
        print(f"[{self.name}] 🛑 Behavior Terminated.")


class ResNetPredictionBehavior_Once(py_trees.behaviour.Behaviour):
    def __init__(self, name="ResNetBehaviorNav", model_path="", class_map=None, camera: RealSenseCamera = None):
        super(ResNetPredictionBehavior_Once, self).__init__(name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load ResNet-18 architecture without pretrained weights
        self.model = models.resnet18(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)
        # _model_path = "W:\gasSensor_ws\GasSensor_ws\src\BT\ztests\man_test_scripts\\resnet18_chemspeed_plate_detection_40_v2.pth"
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        # self.model.to(self.device)
        self.model.eval().to(self.device)

        # self.class_map = class_map if class_map else self.load_class_map()
        self.class_map = self.load_class_map()
        self.camera = camera

        self._blackboard = py_trees.blackboard.Client()
        self._blackboard.register_key(key="ResNet18_best_match", access=py_trees.common.Access.WRITE)
        self._blackboard.register_key(key="ResNet18_scores", access=py_trees.common.Access.WRITE)
        self._blackboard.register_key(key="is_obstacle_detected", access=py_trees.common.Access.WRITE)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def load_class_map(self):
        return ["hazard", "no hazard"]

    def capture_and_save_image(self, image_tensor):
        timestamp = time.strftime("%H%M%S")
        filename = f"resnet_{timestamp}.png"

        image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3)
        image_np = image_np * std.numpy() + mean.numpy()
        image_np = np.clip(image_np * 255, 0, 255).astype('uint8')

        cv2.imwrite(filename, cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
        print(f"Image saved as {filename}")

    def initialise(self):
        if self.camera and not self.camera.is_running:
            self.camera.start_camera()
        self._blackboard.is_obstacle_detected = False

    def update(self):
        try:
            if not self.camera or not self.camera.is_running:
                print(f"[{self.name}] ❌ Camera not available!")
                return py_trees.common.Status.FAILURE

            img = self.camera.capture_image()
            if img is None:
                print(f"[{self.name}] ❌ No valid image captured!")
                return py_trees.common.Status.FAILURE
                
            cv2.imwrite("CS_hazard.jpg", img)
            print(f"Image saved: CS_hazard.jpg")
            time.sleep(1)
            image = Image.open("CS_hazard.jpg").convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(image_tensor)
                _, predicted = outputs.max(1)

            label_text = self.class_map[predicted.item()]

            self._blackboard.ResNet18_best_match = label_text

            print(f"[{self.name}] 🔍 Prediction: {label_text}")

            if label_text == "object_on_floor" or label_text == "hazard":
                print(f"[{self.name}] ⚠️ Obstacle detected!")
                self._blackboard.is_obstacle_detected = True
                # self.capture_and_save_image(image_tensor)
                return py_trees.common.Status.SUCCESS

            else:
                print(f"[{self.name}] ✅ No obstacles detected.")
                self._blackboard.is_obstacle_detected = False
                return py_trees.common.Status.SUCCESS

        except Exception as e:
            print(f"[{self.name}] ❌ Error: {e}")
            return py_trees.common.Status.FAILURE

    def terminate(self, new_status):
        print(f"[{self.name}] 🛑 Behavior Terminated.")
