from transformers import ViTFeatureExtractor, ViTForImageClassification
import torch
import logging
from PIL import Image
import argparse
import os
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_EVAL_OUTPUT_DIR = "./evaluation_results"  # Default output directory from model_evaluation.py
DEFAULT_TEST_DATASET_DIR = "/Users/guanz/Documents/cs229/project/CASIA2.0_revised/resized_splits/test"
TP_TN_COUNT = 20  # Number of TP and TN cases


def load_model(model_path="./results"):
    """Loads the fine-tuned ViT model."""
    logger.info("Loading fine-tuned model...")
    model = ViTForImageClassification.from_pretrained(model_path, output_attentions=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Model loaded on device: {device}")
    return model, device


def preprocess_image(image_path, feature_extractor):
    """Loads and preprocesses a single image."""
    logger.info(f"Preprocessing the image: {image_path}")
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
        label_map = {0: "Authentic", 1: "Tampered"}
        label = label_map.get(prediction, "Unknown")
        attentions = outputs.attentions  # Get the attention maps
        return label, attentions


def visualize_attention(image, attentions, output_dir, prefix):
    """Visualizes the attention maps and saves the output image."""
    logger.info("Visualizing attention maps...")
    attn_weights = attentions[-1]
    attn_weights = attn_weights.mean(dim=1)  # Average over all heads

    cls_attn = attn_weights[0, 0, 1:]  # Exclude CLS token
    num_patches = int(cls_attn.shape[0] ** 0.5)
    cls_attn = cls_attn.reshape(num_patches, num_patches).detach().cpu().numpy()
    cls_attn = cv2.resize(cls_attn, image.size)

    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(image)
    ax.imshow(cls_attn, cmap='jet', alpha=0.5)
    plt.axis('off')

    output_image_path = os.path.join(output_dir, f"{prefix}_attention.png")
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    logger.info(f"Attention map saved to {output_image_path}")


def load_default_inputs(fp_file, fn_file, test_dir):
    """Load default FP, FN, and random TP/TN inputs."""
    logger.info("Loading default FP and FN cases...")
    with open(fp_file, "r") as f:
        fp_list = [line.strip() for line in f.readlines()]

    with open(fn_file, "r") as f:
        fn_list = [line.strip() for line in f.readlines()]

    logger.info(f"Selecting {TP_TN_COUNT} random TP and {TP_TN_COUNT} random TN cases...")
    tp_list = []
    tn_list = []

    for label, folder in [(0, "Au"), (1, "Tp")]:
        class_dir = os.path.join(test_dir, folder)
        all_files = [os.path.join(class_dir, fname) for fname in os.listdir(class_dir)]
        random.shuffle(all_files)
        if label == 0:
            tn_list = all_files[:TP_TN_COUNT]
        else:
            tp_list = all_files[:TP_TN_COUNT]

    return fp_list, fn_list, tp_list + tn_list


def process_images(image_list, feature_extractor, model, device, output_dir, prefix_map):
    """Processes a list of images and generates predictions and attention maps."""
    for image_path in tqdm(image_list, desc="Processing Images"):
        try:
            image_tensor, original_image = preprocess_image(image_path, feature_extractor)
            label, attentions = evaluate_image(model, device, image_tensor)
            logger.info(f"Prediction for {image_path}: {label}")

            prefix = prefix_map[image_path]
            input_image_filename = os.path.basename(image_path)
            renamed_image_filename = f"{prefix}_{input_image_filename}"
            output_image_path = os.path.join(output_dir, renamed_image_filename)
            original_image.save(output_image_path)
            logger.info(f"Original image saved to {output_image_path}")

            visualize_attention(original_image, attentions, output_dir, renamed_image_filename.split('.')[0])

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")


def main(fp_list=None, fn_list=None, correctly_labeled_list=None, output_dir="./output_images"):
    os.makedirs(output_dir, exist_ok=True)

    if not (fp_list and fn_list and correctly_labeled_list):
        fp_list, fn_list, correctly_labeled_list = load_default_inputs(
            fp_file=os.path.join(DEFAULT_MODEL_EVAL_OUTPUT_DIR, "false_positives.txt"),
            fn_file=os.path.join(DEFAULT_MODEL_EVAL_OUTPUT_DIR, "false_negatives.txt"),
            test_dir=DEFAULT_TEST_DATASET_DIR,
        )

    logger.info("Loading feature extractor and model...")
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    model, device = load_model()

    logger.info("Processing all images...")
    all_images = fp_list + fn_list + correctly_labeled_list
    prefix_map = {image: "fp" for image in fp_list}
    prefix_map.update({image: "fn" for image in fn_list})
    prefix_map.update({image: "tp" if "Tp" in image else "tn" for image in correctly_labeled_list})
    process_images(all_images, feature_extractor, model, device, output_dir, prefix_map)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate images with a fine-tuned ViT model and visualize attention maps.")
    parser.add_argument('--fp_list', type=str, nargs='+', help='List of false positive image paths.')
    parser.add_argument('--fn_list', type=str, nargs='+', help='List of false negative image paths.')
    parser.add_argument('--correctly_labeled_list', type=str, nargs='+', help='List of correctly labeled images.')
    parser.add_argument('--output_dir', type=str, default='./output_images', help='Directory to save the output images and attention maps.')
    args = parser.parse_args()

    main(
        fp_list=args.fp_list,
        fn_list=args.fn_list,
        correctly_labeled_list=args.correctly_labeled_list,
        output_dir=args.output_dir,
    )
