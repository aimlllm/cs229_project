# dataset.py

from PIL import Image
import os
from torch.utils.data import Dataset
import torch
from transformers import ViTFeatureExtractor

class CIFakeDataset(Dataset):
    def __init__(self, real_dir, fake_dir, feature_extractor):
        self.real_images = [os.path.join(real_dir, img) for img in os.listdir(real_dir)]
        self.fake_images = [os.path.join(fake_dir, img) for img in os.listdir(fake_dir)]
        self.image_paths = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)
        self.feature_extractor = feature_extractor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        inputs = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)  # Remove extra dimension
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return {'pixel_values': pixel_values, 'labels': label}
