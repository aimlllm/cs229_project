# model_utils.py

from transformers import ViTForImageClassification

def load_model():
    # Load the pre-trained ViT model
    model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")
    model.eval()  # Set model to evaluation mode
    return model
