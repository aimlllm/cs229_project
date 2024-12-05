import os
import json
import torch
import shutil
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from transformers import ViTFeatureExtractor, ViTForImageClassification
from dataset import CASIA2Dataset
from pathlib import Path
from tqdm import tqdm

# Paths
model_path = "./results"  # Path to the fine-tuned model
test_data_path = Path("/Users/guanz/Documents/cs229/project/CASIA2.0_revised/resized_splits/test")
output_path = Path("./evaluation_results")
fp_folder = output_path / "false_positives"
fn_folder = output_path / "false_negatives"
fp_list_file = output_path / "false_positives.txt"
fn_list_file = output_path / "false_negatives.txt"

# Ensure output directories exist
output_path.mkdir(parents=True, exist_ok=True)
fp_folder.mkdir(parents=True, exist_ok=True)
fn_folder.mkdir(parents=True, exist_ok=True)

# Load model and feature extractor
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading model and feature extractor...")
model = ViTForImageClassification.from_pretrained(model_path).to(device)
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# Load test dataset
print("Loading test dataset...")
test_dataset = CASIA2Dataset(data_dir=test_data_path, feature_extractor=feature_extractor)

# Evaluation
print("Evaluating model...")
true_labels = []
pred_labels = []
probabilities = []

# Loop through test dataset
for item in tqdm(test_dataset, desc="Evaluating"):
    pixel_values = item["pixel_values"].unsqueeze(0).to(device)
    true_label = item["labels"]

    with torch.no_grad():
        outputs = model(pixel_values)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy().flatten()
        pred_label = np.argmax(probs)
        probabilities.append(probs[1])  # Probability of being "Tp" (Tampered)

    true_labels.append(true_label)
    pred_labels.append(pred_label)

# Compute evaluation metrics
true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)

accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, pos_label=1)
recall = recall_score(true_labels, pred_labels, pos_label=1)
f1 = f1_score(true_labels, pred_labels, pos_label=1)
conf_matrix = confusion_matrix(true_labels, pred_labels)
roc_auc = roc_auc_score(true_labels, probabilities)

# Save metrics
metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1,
    "Confusion Matrix": conf_matrix.tolist(),
    "AUC-ROC": roc_auc,
}
with open(output_path / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Metrics saved to {output_path / 'metrics.json'}")
print("Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value}")

# Identify false positives and false negatives
fp_list = []
fn_list = []

for idx, (true, pred) in enumerate(zip(true_labels, pred_labels)):
    file_path = test_dataset.data[idx]  # Original file path

    if true == 0 and pred == 1:  # False Positive
        fp_list.append(file_path)
        shutil.copy(file_path, fp_folder / Path(file_path).name)
    elif true == 1 and pred == 0:  # False Negative
        fn_list.append(file_path)
        shutil.copy(file_path, fn_folder / Path(file_path).name)

# Save false positive and false negative lists
with open(fp_list_file, "w") as f:
    f.write("\n".join(fp_list))
with open(fn_list_file, "w") as f:
    f.write("\n".join(fn_list))

print(f"False positives saved to {fp_list_file}")
print(f"False negatives saved to {fn_list_file}")
