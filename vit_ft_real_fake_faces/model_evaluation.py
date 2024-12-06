# model_evaluation.py

from transformers import ViTFeatureExtractor, ViTForImageClassification
from torch.utils.data import DataLoader
from model_utils import load_model
from dataset import RealFakeDataset
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_score, recall_score, \
    f1_score, roc_auc_score, roc_curve
import torch
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths to output directories
OUTPUT_DIR = "./evaluation_results"
HUMAN_EVAL_DIR = os.path.join(OUTPUT_DIR, "human_eval")

# Paths to output FP and FN lists
FP_LIST_FILE = os.path.join(OUTPUT_DIR, "false_positives.txt")
FN_LIST_FILE = os.path.join(OUTPUT_DIR, "false_negatives.txt")

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(HUMAN_EVAL_DIR, exist_ok=True)


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues, output_path="confusion_matrix.png"):
    """
    This function prints and plots the confusion matrix.
    """
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_roc_curve(fpr, tpr, roc_auc, output_path="roc_curve.png"):
    """
    Plot the ROC curve and save to file.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


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
        batch_size=128,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )

    # Load the fine-tuned model
    logger.info("Loading fine-tuned model...")
    model = ViTForImageClassification.from_pretrained("./results")

    # Use MPS if available, else CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Set model to evaluation mode
    model.eval()

    # Evaluate model
    logger.info("Evaluating model on test dataset...")
    all_labels = []
    all_predictions = []
    all_probs = []
    all_file_paths = test_dataset.image_paths

    # Use torch.no_grad() for inference
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['pixel_values'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            outputs = model(inputs)

            # Predictions
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            predictions = logits.argmax(dim=-1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)

    # Compute various metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='binary', pos_label=1)
    recall = recall_score(all_labels, all_predictions, average='binary', pos_label=1)
    f1 = f1_score(all_labels, all_predictions, average='binary', pos_label=1)

    # For AUC-ROC, use probabilities of the positive class (assumed class '1' = Fake)
    roc_auc = roc_auc_score(all_labels, all_probs[:, 1])

    report = classification_report(all_labels, all_predictions, target_names=['Real', 'Fake'])
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    logger.info(f"Model accuracy on RealFake test dataset: {accuracy * 100:.2f}%")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-score: {f1:.4f}")
    logger.info(f"AUC-ROC: {roc_auc:.4f}")
    logger.info(f"\nClassification Report:\n{report}")
    logger.info(f"\nConfusion Matrix:\n{conf_matrix}")

    # Plot and save confusion matrix
    cm_plot_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plot_confusion_matrix(conf_matrix, classes=['Real', 'Fake'], output_path=cm_plot_path)
    logger.info(f"Confusion matrix plot saved to {cm_plot_path}")

    # Plot and save ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs[:, 1])
    roc_plot_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plot_roc_curve(fpr, tpr, roc_auc, output_path=roc_plot_path)
    logger.info(f"ROC curve plot saved to {roc_plot_path}")

    # Identify false positives (FP) and false negatives (FN)
    fp_list = []
    fn_list = []

    for idx, (true_label, pred_label) in enumerate(zip(all_labels, all_predictions)):
        file_path = all_file_paths[idx]
        # true_label: 0=Real, 1=Fake
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

    # Copy the FP and FN files into a separate folder for human evaluation
    fp_human_eval = os.path.join(HUMAN_EVAL_DIR, "false_positives.txt")
    fn_human_eval = os.path.join(HUMAN_EVAL_DIR, "false_negatives.txt")
    shutil.copy(FP_LIST_FILE, fp_human_eval)
    shutil.copy(FN_LIST_FILE, fn_human_eval)
    logger.info(f"False positives and negatives copied to {HUMAN_EVAL_DIR} for human evaluation.")


if __name__ == "__main__":
    main()
