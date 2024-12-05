# dataset.py

from PIL import Image
import os
from torch.utils.data import Dataset
import torch
from transformers import ViTFeatureExtractor


class RealFakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, feature_extractor):
        """
        Dataset for Real and Fake images.
        Args:
            real_dir (str): Path to directory containing real images.
            fake_dir (str): Path to directory containing fake images.
            feature_extractor: Pre-trained feature extractor (e.g., ViTFeatureExtractor).
        """
        self.real_images = [os.path.join(real_dir, img) for img in os.listdir(real_dir)]
        self.fake_images = [os.path.join(fake_dir, img) for img in os.listdir(fake_dir)]
        self.image_paths = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get an image and its label.
        Args:
            idx (int): Index of the image.
        Returns:
            dict: Contains pixel_values (image data), labels (real: 0, fake: 1), and image_path.
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)  # Remove extra dimension
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {'pixel_values': pixel_values, 'labels': label, 'image_path': image_path}
