from PIL import Image
import os
from pathlib import Path

# Base paths
input_base_path = Path("/Users/guanz/Documents/cs229/project/CASIA2.0_revised/splits")
output_base_path = Path("/Users/guanz/Documents/cs229/project/tmp_CASIA2.0_revised/resized_splits")
cropped_output_base_path = Path("/Users/guanz/Documents/cs229/project/tmp_CASIA2.0_revised/cropped_images")

# Resize parameters
TARGET_SIZE = 224


# Resize and crop function
def resize_and_crop(image_path, target_size, cropped_dir, original_dir):
    with Image.open(image_path) as img:
        # Convert to RGB (if needed)
        img = img.convert("RGB")
        original_image_path = original_dir / image_path.name

        # Save the original image for comparison
        img.save(original_image_path)

        # Get dimensions
        width, height = img.size
        cropped_image = img

        # Determine cropping dimensions
        if width > height:
            offset = (width - height) // 2
            cropped_image = img.crop((offset, 0, offset + height, height))  # Crop to square
        elif height > width:
            offset = (height - width) // 2
            cropped_image = img.crop((0, offset, width, offset + width))  # Crop to square

        # Save cropped image before resizing for comparison
        cropped_image_path = cropped_dir / image_path.name
        cropped_image.save(cropped_image_path)

        # Resize to target size
        resized_image = cropped_image.resize((target_size, target_size), Image.LANCZOS)
        return resized_image


# Process images
def process_images(input_dir, output_dir, cropped_dir, target_size):
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
                input_path = Path(root) / file
                relative_path = input_path.relative_to(input_dir)
                output_path = output_dir / relative_path
                original_dir = output_dir.parent / "original"
                cropped_dir = output_dir.parent / "cropped"

                # Ensure output and comparison directories exist
                output_path.parent.mkdir(parents=True, exist_ok=True)
                original_dir.mkdir(parents=True, exist_ok=True)
                cropped_dir.mkdir(parents=True, exist_ok=True)

                # Skip already processed files
                if output_path.exists():
                    print(f"Skipping {output_path}, already exists.")
                    continue

                # Resize and save images
                try:
                    resized_image = resize_and_crop(input_path, target_size, cropped_dir, original_dir)
                    resized_image.save(output_path)
                    print(f"Processed and saved resized: {output_path}")
                    print(f"Saved original: {original_dir / file}")
                    print(f"Saved cropped: {cropped_dir / file}")
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")


# Apply resizing to all splits
for split in ["train", "dev", "test"]:
    for label in ["Au", "Tp"]:
        input_dir = input_base_path / split / label
        output_dir = output_base_path / split / label
        cropped_dir = cropped_output_base_path / split / label
        process_images(input_dir, output_dir, cropped_dir, TARGET_SIZE)

print("Image resizing, cropping, and saving completed!")
