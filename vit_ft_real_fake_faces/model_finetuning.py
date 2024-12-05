from transformers import TrainingArguments, Trainer, ViTFeatureExtractor, TrainerCallback
from model_utils import load_model  # Assuming this contains your model loading logic
from dataset import RealFakeDataset  # Replaced CIFakeDataset with RealFakeDataset
from sklearn.metrics import accuracy_score
import logging
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and feature extractor
logger.info("Loading model and feature extractor...")
model = load_model()
feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

# Set device for M1 Mac
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Training and testing paths
train_real_dir = "/Users/guanz/Documents/cs229/project/real_fake_faces_140k/real_vs_fake/resized/train_sampled/real"
train_fake_dir = "/Users/guanz/Documents/cs229/project/real_fake_faces_140k/real_vs_fake/resized/train_sampled/fake"
test_real_dir = "/Users/guanz/Documents/cs229/project/real_fake_faces_140k/real_vs_fake/resized/validation_sampled/real"
test_fake_dir = "/Users/guanz/Documents/cs229/project/real_fake_faces_140k/real_vs_fake/resized/validation_sampled/fake"

# Load datasets
logger.info("Loading training and testing datasets...")
train_dataset = RealFakeDataset(real_dir=train_real_dir, fake_dir=train_fake_dir, feature_extractor=feature_extractor)
test_dataset = RealFakeDataset(real_dir=test_real_dir, fake_dir=test_fake_dir, feature_extractor=feature_extractor)

# Define accuracy computation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc}

# Training arguments for optimization on M1 Mac
logger.info("Setting up training arguments...")
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=32,     # Adjust based on memory availability
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    bf16=True,                          # Use bfloat16 precision if supported, otherwise remove this
    gradient_accumulation_steps=4,      # Simulates a larger batch size
    learning_rate=5e-5,
    evaluation_strategy="epoch",        # Evaluate only at the end of each epoch
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=50,                   # Log every 50 steps
    dataloader_num_workers=4,           # Speed up data loading
)

# Trainer setup
logger.info("Initializing Trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# Training with custom logging
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logger.info(f"Step {state.global_step} - Training metrics: {logs}")

# Add the logging callback
trainer.add_callback(LoggingCallback())

if __name__ == "__main__":
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")
    trainer.save_model()
    logger.info("Model saved.")
