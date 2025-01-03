from PIL import Image
import os
from pathlib import Path

# Base paths
input_base_path = Path("/Users/guanz/Documents/cs229/project/CASIA2.0_revised/splits")
output_base_path = Path("/Users/guanz/Documents/cs229/project/CASIA2.0_revised_no_crop/resized_splits")

# Resize parameters
TARGET_SIZE = 224


# Resize function
def resize_image(image_path, target_size):
    with Image.open(image_path) as img:
        # Convert to RGB
        img = img.convert("RGB")

        # Resize to target size without cropping
        resized_image = img.resize((target_size, target_size), Image.LANCZOS)
        return resized_image


# Process images
def process_images(input_dir, output_dir, target_size):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                input_path = Path(root) / file
                relative_path = input_path.relative_to(input_dir)
                output_path = output_dir / relative_path

                # Ensure output directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Skip already processed files
                if output_path.exists():
                    print(f"Skipping {output_path}, already exists.")
                    continue

                # Resize and save images
                try:
                    resized_image = resize_image(input_path, target_size)
                    resized_image.save(output_path)
                    print(f"Processed and saved resized: {output_path}")
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")


# Apply resizing to all splits
for split in ["train", "dev", "test"]:
    for label in ["Au", "Tp"]:
        input_dir = input_base_path / split / label
        output_dir = output_base_path / split / label
        process_images(input_dir, output_dir, TARGET_SIZE)

print("Image resizing completed!")
