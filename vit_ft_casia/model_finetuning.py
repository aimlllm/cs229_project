from transformers import TrainingArguments, Trainer, ViTImageProcessor, TrainerCallback
from model_utils import load_model
from dataset import CASIA2Dataset
from sklearn.metrics import accuracy_score
import logging
import torch
import psutil
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and image processor
logger.info("Loading model and image processor...")
model = load_model()
image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# Set device for M1 Mac
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
model.to(device)

# Paths to the resized dataset
data_base_path = Path("/Users/guanz/Documents/cs229/project/CASIA2.0_resized_224_no_crop/resized_splits")
train_dir = data_base_path / "train"
dev_dir = data_base_path / "dev"
test_dir = data_base_path / "test"

# Load datasets
logger.info("Loading training, validation, and testing datasets...")
train_dataset = CASIA2Dataset(data_dir=train_dir, feature_extractor=image_processor)
dev_dataset = CASIA2Dataset(data_dir=dev_dir, feature_extractor=image_processor)
test_dataset = CASIA2Dataset(data_dir=test_dir, feature_extractor=image_processor)
logger.info(f"Train dataset size: {len(train_dataset)}")
logger.info(f"Validation dataset size: {len(dev_dataset)}")

# Define accuracy computation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# Training arguments
logger.info("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,  # Reduced batch size to avoid memory issues
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    gradient_accumulation_steps=4,  # Adjusted for smaller effective batch size
    learning_rate=3e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=20,
    dataloader_num_workers=8,
    bf16=False,  # Disabled bf16 due to MPS backend limitations
    fp16=False,  # Ensure compatibility with MPS if using CPU/GPU fallback
)

# Trainer setup
logger.info("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
)

# Custom logging callback
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

# Add the verbose logging callback
trainer.add_callback(VerboseLoggingCallback())

if __name__ == "__main__":
    logger.info("Starting fine-tuning...")
    trainer.train()
    logger.info("Fine-tuning complete. Saving the model...")
    trainer.save_model()
    logger.info("Model saved successfully.")
