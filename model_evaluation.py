# model_evaluation.py

from transformers import ViTFeatureExtractor, ViTForImageClassification  # Import ViTForImageClassification
from torch.utils.data import DataLoader
from model_utils import load_model  # Assuming this contains your model loading logic
from dataset import CIFakeDataset  # Assuming this contains your dataset loading logic
from sklearn.metrics import classification_report
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Paths to test dataset
    test_real_dir = "/Users/guanz/Documents/cs229/project/CIFake/test/REAL"
    test_fake_dir = "/Users/guanz/Documents/cs229/project/CIFake/test/FAKE"

    # Load feature extractor and test dataset
    logger.info("Loading feature extractor and test dataset...")
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    test_dataset = CIFakeDataset(real_dir=test_real_dir, fake_dir=test_fake_dir, feature_extractor=feature_extractor)
    dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load the fine-tuned model
    logger.info("Loading fine-tuned model...")
    model = ViTForImageClassification.from_pretrained("./results")  # Fine-tuned model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Evaluate model
    logger.info("Evaluating model on test dataset...")
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(inputs)
            predictions = outputs.logits.argmax(dim=-1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    # Accuracy calculation and detailed report
    accuracy = sum(1 for x, y in zip(all_labels, all_predictions) if x == y) / len(all_labels)
    logger.info(f"Model accuracy on CIFake test dataset: {accuracy * 100:.2f}%")
    report = classification_report(all_labels, all_predictions, target_names=['Real', 'Fake'])
    logger.info(f"\n{report}")

if __name__ == "__main__":
    main()
