# model_evaluation.py

from transformers import ViTFeatureExtractor, ViTForImageClassification
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
    test_real_dir = "/Users/guanz/Documents/cs229/project/real_fake_faces_140k/real_vs_fake/resized/test/real"
    test_fake_dir = "/Users/guanz/Documents/cs229/project/real_fake_faces_140k/real_vs_fake/resized/test/fake"
    
    # Load feature extractor and test dataset
    logger.info("Loading feature extractor and test dataset...")
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    test_dataset = CIFakeDataset(real_dir=test_real_dir, fake_dir=test_fake_dir, feature_extractor=feature_extractor)
    
    # Increase batch size and set pin_memory for faster data loading
    dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=4)

    # Load the original pre-trained model
    logger.info("Loading original pre-trained model...")
    model = ViTForImageClassification.from_pretrained("./results")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Evaluate model
    logger.info("Evaluating model on test dataset...")
    all_labels = []
    all_predictions = []

    # Use torch.no_grad() for faster inference by disabling gradient calculation
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['pixel_values'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
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
