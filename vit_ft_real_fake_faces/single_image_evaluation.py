# single_image_evaluation.py

from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch
import logging
from PIL import Image
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(model_path="./results"):
    """Loads the fine-tuned ViT model."""
    logger.info("Loading fine-tuned model...")
    model = ViTForImageClassification.from_pretrained(model_path)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    return model, device

def preprocess_image(image_path, feature_extractor):
    """Loads and preprocesses a single image."""
    logger.info("Preprocessing the image...")
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs['pixel_values']

def evaluate_image(model, device, image_tensor):
    """Evaluates the model on a single image tensor."""
    model.eval()
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        prediction = outputs.logits.argmax(dim=-1).item()
        label = "Real" if prediction == 0 else "Fake"
        return label

def main(image_path):
    # Load feature extractor
    logger.info("Loading feature extractor...")
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

    # Preprocess the image
    image_tensor = preprocess_image(image_path, feature_extractor)

    # Load model
    model, device = load_model()

    # Evaluate the image
    logger.info("Evaluating the image...")
    label = evaluate_image(model, device, image_tensor)
    logger.info(f"The model prediction for the image is: {label}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        logger.error("Usage: python single_image_evaluation.py <image_path>")
    else:
        main(sys.argv[1])
