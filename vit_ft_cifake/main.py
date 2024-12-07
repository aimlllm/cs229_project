# main.py

from torch.utils.data import DataLoader
from transformers import ViTFeatureExtractor
from dataset import CIFakeDataset
from model_evaluation import load_model, evaluate_model
from sklearn.metrics import classification_report
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    real_dir = "/Users/guanz/Documents/cs229/project/CIFake/test/REAL"
    fake_dir = "/Users/guanz/Documents/cs229/project/CIFake/test/FAKE"
    # real_dir = "/Users/guanz/Documents/cs229/project/CIFake/test/small/real"
    # fake_dir = "/Users/guanz/Documents/cs229/project/CIFake/test/small/fake"

    logger.info("Loading feature extractor...")
    feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")

    logger.info("Preparing dataset and dataloader...")
    dataset = CIFakeDataset(real_dir=real_dir, fake_dir=fake_dir, feature_extractor=feature_extractor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    logger.info(f"Dataset loaded with {len(dataset)} samples.")

    logger.info("Loading model...")
    model = load_model()

    logger.info("Evaluating model...")
    accuracy, all_labels, all_predictions = evaluate_model(model, dataloader)
    logger.info(f"Model accuracy on CIFake dataset: {accuracy * 100:.2f}%")

    logger.info("Generating detailed classification report...")
    report = classification_report(all_labels, all_predictions, target_names=['Real', 'Fake'])
    logger.info(f"\n{report}")

if __name__ == "__main__":
    main()
