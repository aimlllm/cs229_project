from transformers import ViTFeatureExtractor, ViTForImageClassification
from torch.utils.data import DataLoader
from model_utils import load_model
from dataset import RealFakeDataset
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import torch
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths to output FP and FN lists
OUTPUT_DIR = "./evaluation_results"
FP_LIST_FILE = os.path.join(OUTPUT_DIR, "false_positives.txt")
FN_LIST_FILE = os.path.join(OUTPUT_DIR, "false_negatives.txt")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    # Paths to test dataset
    test_real_dir = "/Users/guanz/Documents/cs229/project/real_fake_faces_140k/real_vs_fake/resized/test/real"
    test_fake_dir = "/Users/guanz/Documents/cs229/project/real_fake_faces_140k/real_vs_fake/resized/test/fake"

    # Load feature extractor and test dataset
    logger.info("Loading feature extractor and test dataset...")
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    test_dataset = RealFakeDataset(real_dir=test_real_dir, fake_dir=test_fake_dir, feature_extractor=feature_extractor)

    # Optimize DataLoader for faster data loading
    dataloader = DataLoader(
        test_dataset,
        batch_size=128,  # Larger batch size to utilize more memory
        shuffle=False,
        pin_memory=True,
        num_workers=8  # Use more workers to leverage the 10-core CPU
    )

    # Load the fine-tuned model
    logger.info("Loading fine-tuned model...")
    model = ViTForImageClassification.from_pretrained("./results")

    # Use MPS (Metal Performance Shaders) for GPU acceleration on MacBook Pro
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Set model to evaluation mode
    model.eval()

    # Evaluate model
    logger.info("Evaluating model on test dataset...")
    all_labels = []
    all_predictions = []
    all_file_paths = test_dataset.image_paths  # Corrected to use .image_paths

    # Use torch.no_grad() for faster inference by disabling gradient calculation
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['pixel_values'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            outputs = model(inputs)
            predictions = outputs.logits.argmax(dim=-1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Compute accuracy and other metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions, target_names=['Real', 'Fake'])
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    logger.info(f"Model accuracy on RealFake test dataset: {accuracy * 100:.2f}%")
    logger.info(f"\nClassification Report:\n{report}")
    logger.info(f"\nConfusion Matrix:\n{conf_matrix}")

    # Identify false positives (FP) and false negatives (FN)
    fp_list = []
    fn_list = []

    for idx, (true_label, pred_label) in enumerate(zip(all_labels, all_predictions)):
        file_path = all_file_paths[idx]

        if true_label == 0 and pred_label == 1:  # False Positive
            fp_list.append(file_path)
        elif true_label == 1 and pred_label == 0:  # False Negative
            fn_list.append(file_path)

    # Save FP and FN lists to text files
    with open(FP_LIST_FILE, "w") as fp_file:
        fp_file.write("\n".join(fp_list))
    with open(FN_LIST_FILE, "w") as fn_file:
        fn_file.write("\n".join(fn_list))

    logger.info(f"False positives saved to {FP_LIST_FILE}")
    logger.info(f"False negatives saved to {FN_LIST_FILE}")


if __name__ == "__main__":
    main()
