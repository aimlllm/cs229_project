import os
import json
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt
import logging
import psutil
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from transformers import (
    TrainingArguments,
    Trainer,
    ViTImageProcessor,
    TrainerCallback,
    ViTModel
)
import torch.nn as nn
from dataset import CASIA2Dataset
from tqdm import tqdm

# -----------------------------------------
# Logging setup
# -----------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------
# Model Definition
# -----------------------------------------
class ViTBinaryClassifier(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224-in21k", num_classes=2):
        super(ViTBinaryClassifier, self).__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        pooled_output = outputs.pooler_output  # [CLS] token representation
        logits = self.classifier(pooled_output)
        return logits

def load_model():
    # Loads and returns the ViTBinaryClassifier model
    model = ViTBinaryClassifier("google/vit-base-patch16-224-in21k", num_classes=2)
    return model

# -----------------------------------------
# Paths and directories
# -----------------------------------------
data_base_path = Path("/Users/guanz/Documents/cs229/project/CASIA2.0_resized_224_no_crop/resized_splits")
train_dir = data_base_path / "train"
dev_dir = data_base_path / "dev"
test_dir = data_base_path / "test"

output_path = Path("./linear_head_evaluation_results")
fp_folder = output_path / "false_positives"
fn_folder = output_path / "false_negatives"
fp_list_file = output_path / "false_positives.txt"
fn_list_file = output_path / "false_negatives.txt"
confusion_matrix_file = output_path / "confusion_matrix.png"
roc_curve_file = output_path / "roc_curve.png"

output_path.mkdir(parents=True, exist_ok=True)
fp_folder.mkdir(parents=True, exist_ok=True)
fn_folder.mkdir(parents=True, exist_ok=True)

# -----------------------------------------
# Load model and image processor
# -----------------------------------------
logger.info("Loading model and image processor...")
model = load_model()

# Freeze ViT layers (only train the classification head)
for param in model.vit.parameters():
    param.requires_grad = False

image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Device selection
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
model.to(device)

# -----------------------------------------
# Load datasets
# -----------------------------------------
logger.info("Loading training, validation, and testing datasets...")
train_dataset = CASIA2Dataset(data_dir=train_dir, feature_extractor=image_processor)
dev_dataset = CASIA2Dataset(data_dir=dev_dir, feature_extractor=image_processor)
test_dataset = CASIA2Dataset(data_dir=test_dir, feature_extractor=image_processor)

logger.info(f"Train dataset size: {len(train_dataset)}")
logger.info(f"Validation dataset size: {len(dev_dataset)}")
logger.info(f"Test dataset size: {len(test_dataset)}")

# -----------------------------------------
# Metrics for Trainer
# -----------------------------------------
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# -----------------------------------------
# Training arguments
# -----------------------------------------
logger.info("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    gradient_accumulation_steps=4,
    learning_rate=3e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=20,
    dataloader_num_workers=8,
    bf16=False,
    fp16=False,
)

# -----------------------------------------
# Custom logging callback
# -----------------------------------------
class VerboseLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            loss = logs.get("loss", "N/A")
            learning_rate = logs.get("learning_rate", "N/A")
            step = state.global_step
            logger.info(f"[Step {step}] Loss: {loss}, Learning Rate: {learning_rate}")

            # Log system resource usage
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            logger.info(f"CPU Usage: {cpu_usage}% | Memory Usage: {memory_usage}%")

    def on_train_epoch_end(self, args, state, control, **kwargs):
        logger.info(f"Epoch {state.epoch} completed. Evaluating...")

# -----------------------------------------
# Trainer setup
# -----------------------------------------
logger.info("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
)
trainer.add_callback(VerboseLoggingCallback())

# -----------------------------------------
# Training and Saving
# -----------------------------------------
if __name__ == "__main__":
    logger.info("Starting fine-tuning with frozen ViT layers...")
    trainer.train()
    logger.info("Fine-tuning complete. Saving the model...")
    trainer.save_model()
    logger.info("Model saved successfully.")

    # -------------------------------------
    # Evaluation on Test Set
    # -------------------------------------
    logger.info("Evaluating model on test set...")
    model.eval()

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
            probabilities.append(probs[1])  # Probability of "Tp" class

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

    logger.info(f"Metrics saved to {output_path / 'metrics.json'}")
    logger.info("Metrics:")
    for key, value in metrics.items():
        logger.info(f"{key}: {value}")

    # Plot and save confusion matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Nt', 'Tp'], rotation=45)
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
    logger.info(f"Confusion matrix plot saved to {confusion_matrix_file}")

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
    logger.info(f"ROC curve plot saved to {roc_curve_file}")

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

    # Save false positive and negative lists
    with open(fp_list_file, "w") as f:
        f.write("\n".join(fp_list))
    with open(fn_list_file, "w") as f:
        f.write("\n".join(fn_list))

    logger.info(f"False positives saved to {fp_list_file}")
    logger.info(f"False negatives saved to {fn_list_file}")
    logger.info("Evaluation complete.")
