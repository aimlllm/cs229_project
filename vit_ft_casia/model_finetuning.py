from transformers import TrainingArguments, Trainer, ViTFeatureExtractor, TrainerCallback
from model_utils import load_model  # Assuming this contains your model loading logic
from dataset import CASIA2Dataset  # Updated dataset logic for your use case
from sklearn.metrics import accuracy_score
import logging
import torch
import psutil  # To monitor system resource usage
import GPUtil  # For GPU resource monitoring (optional)
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and feature extractor
logger.info("Loading model and feature extractor...")
model = load_model()
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# Set device for M1 Mac
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
model.to(device)

# Paths to the resized dataset
data_base_path = Path("/Users/guanz/Documents/cs229/project/CASIA2.0_revised/resized_splits")
train_dir = data_base_path / "train"
dev_dir = data_base_path / "dev"
test_dir = data_base_path / "test"

# Load datasets
logger.info("Loading training, validation, and testing datasets...")
train_dataset = CASIA2Dataset(data_dir=train_dir, feature_extractor=feature_extractor)
dev_dataset = CASIA2Dataset(data_dir=dev_dir, feature_extractor=feature_extractor)
test_dataset = CASIA2Dataset(data_dir=test_dir, feature_extractor=feature_extractor)
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
    per_device_train_batch_size=64,  # Increased batch size to leverage unified memory
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    bf16=True,  # Optimized for speed and precision on M1 GPU
    gradient_accumulation_steps=2,  # Faster updates with larger effective batch size
    learning_rate=3e-5,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=20,
    dataloader_num_workers=8,  # Use more CPU cores for data loading
)

# Trainer setup
logger.info("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,  # Use dev for evaluation
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
            if torch.cuda.is_available():
                gpu_usage = GPUtil.getGPUs()[0].load * 100
                gpu_memory = GPUtil.getGPUs()[0].memoryUsed
                logger.info(f"GPU Usage: {gpu_usage:.2f}% | GPU Memory: {gpu_memory:.2f} MB")

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