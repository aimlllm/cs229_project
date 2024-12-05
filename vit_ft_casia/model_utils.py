# model_utils.py
from transformers import ViTForImageClassification

def load_model():
    """
    Load and configure the ViT model for fine-tuning on the CASIA2 dataset.
    The model is adapted for binary classification: 0 (Authentic) and 1 (Tampered).
    """
    # Load the pre-trained ViT model
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=2,  # Binary classification: Authentic (0) or Tampered (1)
        id2label={0: "Authentic", 1: "Tampered"},  # Map label IDs to names
        label2id={"Authentic": 0, "Tampered": 1}  # Map names to label IDs
    )

    # Set the model to training mode
    model.train()

    return model
