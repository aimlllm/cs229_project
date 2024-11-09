# model_evaluation_curator_fake.py

from transformers import ViTFeatureExtractor, ViTForImageClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report
import torch
import logging
import os
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FakeImageDataset(Dataset):
    def __init__(self, fake_dir, feature_extractor):
        self.fake_dir = fake_dir
        self.feature_extractor = feature_extractor
        self.fake_images = [
            os.path.join(fake_dir, f) for f in os.listdir(fake_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
        ]
    
    def __len__(self):
        return len(self.fake_images)
    
    def __getitem__(self, idx):
        image_path = self.fake_images[idx]
        image = Image.open(image_path).convert("RGB")
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        return {
            "pixel_values": inputs["pixel_values"].squeeze(),
            "labels": torch.tensor(1),  # Label '1' for "Fake"
            "file_name": os.path.basename(image_path)  # Add file name for reference
        }

def main():
    # Path to the FAKE images directory
    fake_dir = "/Users/guanz/Documents/cs229/project/CIFake/resized/curator/REAL"

    # Load feature extractor and test dataset
    logger.info("Loading feature extractor and test dataset...")
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    test_dataset = FakeImageDataset(fake_dir=fake_dir, feature_extractor=feature_extractor)
    dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Load the fine-tuned model
    logger.info("Loading fine-tuned model...")
    model = ViTForImageClassification.from_pretrained("./results")  # Path to fine-tuned model
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Evaluate model and print predictions with file names
    logger.info("Evaluating model on FAKE images...")
    model.eval()
    all_labels = []
    all_predictions = []
    file_names = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(inputs)
            predictions = outputs.logits.argmax(dim=-1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            file_names.extend(batch['file_name'])  # Collect file names

    # Print predictions with file names
    for file_name, prediction in zip(file_names, all_predictions):
        pred_label = "Fake" if prediction == 1 else "Real"
        logger.info(f"File: {file_name}, Prediction: {pred_label}")

    # Accuracy calculation and detailed report
    accuracy = sum(1 for x, y in zip(all_labels, all_predictions) if x == y) / len(all_labels)
    logger.info(f"Model accuracy on FAKE images dataset: {accuracy * 100:.2f}%")
    
    # Adjusted classification report to specify `labels` parameter
    report = classification_report(
        all_labels,
        all_predictions,
        labels=[1],  # Only include the "Fake" class
        target_names=['Fake']
    )
    logger.info(f"\n{report}")

if __name__ == "__main__":
    main()
