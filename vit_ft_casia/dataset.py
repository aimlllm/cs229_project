import os
from PIL import Image
from torch.utils.data import Dataset

class CASIA2Dataset(Dataset):
    def __init__(self, data_dir, feature_extractor):
        """
        Initialize the dataset.
        Args:
            data_dir (str): Path to the dataset directory.
            feature_extractor: Pretrained feature extractor for ViT.
        """
        self.data_dir = data_dir
        self.feature_extractor = feature_extractor
        self.data = []
        self.labels = []

        # Load images and labels
        for label, label_dir in enumerate(["Au", "Tp"]):  # 0 for Au (authentic), 1 for Tp (tampered)
            class_dir = os.path.join(data_dir, label_dir)
            for image_file in os.listdir(class_dir):
                self.data.append(os.path.join(class_dir, image_file))
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single data point.
        Args:
            idx (int): Index of the data point.
        Returns:
            dict: Contains pixel_values and labels for the model.
        """
        image_path = self.data[idx]
        label = self.labels[idx]

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        encoding = self.feature_extractor(images=image, return_tensors="pt")

        # Extract pixel_values and squeeze batch dimension
        pixel_values = encoding["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": label,
        }
