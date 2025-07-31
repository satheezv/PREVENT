import torch
import open_clip
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import numpy as np
import cv2


class CLIPUtility:
    """
    Utility class for using CLIP ViT-L/14 for image-text matching.
    """

    def __init__(self, model_name="ViT-L-14-quickgelu", pretrained="openai"):


        # print(f"[CLIPUtility] 🚀 Initializing CLIP with model: {model_name}"
        #       f"{' (pretrained)' if pretrained else ''}")
        

        finetuned = True
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("W:\gasSensor_ws\GasSensor_ws\src\open_clip\clip_finetuned_final").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("W:\gasSensor_ws\GasSensor_ws\src\open_clip\clip_finetuned_final")
        print(f"[CLIPUtility] 🚀 Initializing CLIP with model: {self.model}"
              f"{' (finetuned)' if finetuned else ''}")
        # values = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)

        # if len(values) == 2:
        #     self.model, self.preprocess = values
        # elif len(values) == 3:
        #     self.model, self.preprocess, _ = values  # Ignore the third value
        # else:
        #     raise ValueError(f"Unexpected number of return values from create_model_and_transforms(): {len(values)}")

        # self.tokenizer = open_clip.get_tokenizer(model_name)
        # self.model.to(self.device)

    def preprocess_image(self, image):
        """
        Converts a NumPy image (from RealSense) into a PIL Image and preprocesses it for CLIP.

        :param image: NumPy array (RealSense image)
        :return: Preprocessed image tensor
        """
        try:
            # Convert NumPy array to PIL Image
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            return self.preprocess(pil_image).unsqueeze(0).to(self.device)
        except Exception as e:
            print(f"[CLIPUtility] ❌ Error in preprocessing image: {e}")
            return None

    def extract_image_features(self, image):
        """
        Extracts image embeddings from CLIP.

        :param image: NumPy image array (directly from RealSense camera)
        :return: Torch tensor with image features.
        """
        try:
            image_tensor = self.preprocess_image(image)
            if image_tensor is None:
                return None

            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
            return image_features.cpu().numpy()
        except Exception as e:
            print(f"[CLIPUtility] ❌ Error extracting image features: {e}")
            return None

    def match_image_to_text(self, image, text_labels):
        """
        Matches an image to a list of text descriptions using CLIP.

        :param image: NumPy image array (directly from RealSense camera)
        :param text_labels: List of textual descriptions.
        :return: Best-matching text and confidence score.
        """
        try:
            inputs = self.processor(images=image, text=text_labels, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)

            pred_idx = probs.argmax(dim=1).item()
            pred_label = text_labels[pred_idx]
            pred_conf = probs[0][pred_idx].item()

            return pred_label, pred_conf

        except Exception as e:
            print(f"[CLIPUtility] ❌ Error in image-text matching: {e}")
            return None, []

