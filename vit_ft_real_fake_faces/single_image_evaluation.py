# single_image_evaluation.py

from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch
import logging
from PIL import Image
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import argparse  # Added argparse for command-line argument parsing

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(model_path="./results"):
    """Loads the fine-tuned ViT model."""
    logger.info("Loading fine-tuned model...")
    model = ViTForImageClassification.from_pretrained(model_path, output_attentions=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, device


def preprocess_image(image_path, feature_extractor):
    """Loads and preprocesses a single image."""
    logger.info("Preprocessing the image...")
    image = Image.open(image_path).convert("RGB")
    inputs = feature_extractor(images=image, return_tensors="pt")
    return inputs['pixel_values'], image


def evaluate_image(model, device, image_tensor):
    """Evaluates the model on a single image tensor and returns attention maps."""
    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        prediction = outputs.logits.argmax(dim=-1).item()
        label_map = {0: "Real", 1: "Fake"}
        label = label_map.get(prediction, "Unknown")
        attentions = outputs.attentions  # Get the attention maps
        return label, attentions


def visualize_attention(image, attentions, output_dir, input_image_filename):
    """Visualizes the attention maps and saves the output image."""
    logger.info("Visualizing attention maps...")
    # Get attention maps from the last layer
    attn_weights = attentions[-1]  # Last layer's attention
    # Average over all heads
    attn_weights = attn_weights.mean(dim=1)  # Shape: [batch_size, seq_len, seq_len]

    # Get the attention weights for the CLS token with respect to other tokens
    cls_attn = attn_weights[0, 0, 1:]  # Exclude CLS token
    num_patches = int(cls_attn.shape[0] ** 0.5)
    # Reshape to 2D spatial attention map
    cls_attn = cls_attn.reshape(num_patches, num_patches).detach().cpu().numpy()
    # Resize attention map to image size
    cls_attn = cv2.resize(cls_attn, image.size)
    # Normalize
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())

    # Plotting
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)
    ax.imshow(cls_attn, cmap='jet', alpha=0.5)
    plt.axis('off')

    # Save the figure
    output_image_path = os.path.join(output_dir, os.path.splitext(input_image_filename)[0] + '_attention.png')
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    logger.info(f"Attention map saved to {output_image_path}")


def main(image_path, output_dir):
    # Load feature extractor
    logger.info("Loading feature extractor...")
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

    # Preprocess the image
    image_tensor, original_image = preprocess_image(image_path, feature_extractor)

    # Load model
    model, device = load_model()

    # Evaluate the image and get attention maps
    logger.info("Evaluating the image...")
    label, attentions = evaluate_image(model, device, image_tensor)
    logger.info(f"The model prediction for the image is: {label}")
    print(f"The model prediction for the image is: {label}")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save the original image to the output directory
    input_image_filename = os.path.basename(image_path)
    output_image_path = os.path.join(output_dir, input_image_filename)
    original_image.save(output_image_path)
    logger.info(f"Input image saved to {output_image_path}")

    # Visualize attention and save the attention map
    visualize_attention(original_image, attentions, output_dir, input_image_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify an image and visualize attention map.")
    parser.add_argument('image_path', type=str, help='Path to the input image.')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save the output images.')
    args = parser.parse_args()

    main(args.image_path, args.output_dir)
