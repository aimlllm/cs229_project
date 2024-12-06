import os
import json
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from transformers import ViTFeatureExtractor, ViTModel
from dataset import CASIA2Dataset
from pathlib import Path
from tqdm import tqdm
import torch.nn as nn

# Paths
model_path = "./results"
test_data_path = Path("/Users/guanz/Documents/cs229/project/CASIA2.0_resized_224_no_crop/resized_splits/test")
output_path = Path("./linear_head_evaluation_results")
fp_folder = output_path / "false_positives"
fn_folder = output_path / "false_negatives"
fp_list_file = output_path / "false_positives.txt"
fn_list_file = output_path / "false_negatives.txt"
confusion_matrix_file = output_path / "confusion_matrix.png"
roc_curve_file = output_path / "roc_curve.png"

# Ensure output directories exist
output_path.mkdir(parents=True, exist_ok=True)
fp_folder.mkdir(parents=True, exist_ok=True)
fn_folder.mkdir(parents=True, exist_ok=True)

# Define a custom ViT-based binary classifier
class ViTBinaryClassifier(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224-in21k", num_classes=2):
        super(ViTBinaryClassifier, self).__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        # We can use the pooled_output directly (it comes from the [CLS] token):
        pooled_output = outputs.pooler_output  # shape: (batch_size, hidden_size)
        logits = self.classifier(pooled_output)
        return logits

# Detect device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading feature extractor...")
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# Initialize model
print("Initializing model...")
model = ViTBinaryClassifier("google/vit-base-patch16-224-in21k")
model.to(device)

# Load the trained weights (assuming 'model_path' contains 'pytorch_model.bin' or a similar checkpoint)
checkpoint_path = os.path.join(model_path, "pytorch_model.bin")
if os.path.exists(checkpoint_path):
    print(f"Loading model weights from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Model weights loaded.")
else:
    raise FileNotFoundError(f"No model checkpoint found at {checkpoint_path}")

model.eval()

# Load test dataset
print("Loading test dataset...")
test_dataset = CASIA2Dataset(data_dir=test_data_path, feature_extractor=feature_extractor)

# Evaluation
print("Evaluating model...")
true_labels = []
pred_labels = []
probabilities = []

for item in tqdm(test_dataset, desc="Evaluating"):
    pixel_values = item["pixel_values"].unsqueeze(0).to(device)
    true_label = item["labels"]

    with torch.no_grad():
        logits = model(pixel_values)
        probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
        pred_label = np.argmax(probs)
        probabilities.append(probs[1])  # Probability of "Tp" (Tampered) class

    true_labels.append(true_label)
    pred_labels.append(pred_label)

# Convert to numpy arrays
true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)

# Compute evaluation metrics
accuracy = accuracy_score(true_labels, pred_labels)
precision = precision_score(true_labels, pred_labels, pos_label=1)
recall = recall_score(true_labels, pred_labels, pos_label=1)
f1 = f1_score(true_labels, pred_labels, pos_label=1)
conf_matrix = confusion_matrix(true_labels, pred_labels)
roc_auc = roc_auc_score(true_labels, probabilities)

# Save metrics as JSON
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

# Plot and save confusion matrix
plt.figure(figsize=(6, 5))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Nt', 'Tp'], rotation=45)  # Adjust class names if necessary
plt.yticks(tick_marks, ['Nt', 'Tp'])

thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.savefig(confusion_matrix_file)
plt.close()

print(f"Confusion matrix plot saved to {confusion_matrix_file}")

# Plot and save ROC curve
fpr, tpr, _ = roc_curve(true_labels, probabilities, pos_label=1)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig(roc_curve_file)
plt.close()

print(f"ROC curve plot saved to {roc_curve_file}")

# Identify false positives and false negatives
fp_list = []
fn_list = []

for idx, (true, pred) in enumerate(zip(true_labels, pred_labels)):
    file_path = test_dataset.data[idx]  # Path to the original image

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
print("Evaluation complete.")
