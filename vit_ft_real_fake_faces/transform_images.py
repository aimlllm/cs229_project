#  transform_images.py

from PIL import Image, ImageOps
import os
from multiprocessing import Pool, cpu_count
import time

def process_image(args):
    img_path, target_dir, size = args
    try:
        img = Image.open(img_path).convert("RGB")
        width, height = img.size

        # Calculate padding to make the image square
        if width > height:
            padding = (width - height) // 2
            img = ImageOps.expand(img, (0, padding, 0, padding), fill=(0, 0, 0))  # Add padding to top and bottom
        elif height > width:
            padding = (height - width) // 2
            img = ImageOps.expand(img, (padding, 0, padding, 0), fill=(0, 0, 0))  # Add padding to left and right

        # Resize to the target size using a high-quality filter
        img = img.resize(size, Image.LANCZOS)
        
        # Save the image in PNG format
        output_name = os.path.splitext(os.path.basename(img_path))[0] + ".png"
        img.save(os.path.join(target_dir, output_name), format="PNG")
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

def resize_images(source_dir, target_dir, size=(224, 224)):
    os.makedirs(target_dir, exist_ok=True)
    # Prepare arguments for parallel processing
    args = [(os.path.join(source_dir, img_name), target_dir, size) for img_name in os.listdir(source_dir)]
    
    # Use multiprocessing Pool to process images in parallel
    with Pool(cpu_count()) as pool:
        pool.map(process_image, args)

if __name__ == "__main__":
    # Paths for original and target resized images
    start_time = time.time()
    
    train_real_dir = "/Users/guanz/Documents/cs229/project/real_fake_faces_140k/real_vs_fake/train_sampled/real"
    train_fake_dir = "/Users/guanz/Documents/cs229/project/real_fake_faces_140k/real_vs_fake/train_sampled/fake"
    
    val_real_dir = "/Users/guanz/Documents/cs229/project/real_fake_faces_140k/real_vs_fake/validation_sampled/real"
    val_fake_dir = "/Users/guanz/Documents/cs229/project/real_fake_faces_140k/real_vs_fake/validation_sampled/fake"

    test_real_dir = "/Users/guanz/Documents/cs229/project/real_fake_faces_140k/real_vs_fake/test/real"
    test_fake_dir = "/Users/guanz/Documents/cs229/project/real_fake_faces_140k/real_vs_fake/test/fake"
    
    resize_images(test_real_dir, "/Users/guanz/Documents/cs229/project/real_fake_faces_140k/real_vs_fake/resized/test/real")
    resize_images(test_fake_dir, "/Users/guanz/Documents/cs229/project/real_fake_faces_140k/real_vs_fake/resized/test/fake")

    resize_images(train_real_dir, "/Users/guanz/Documents/cs229/project/real_fake_faces_140k/real_vs_fake/resized/train_sampled/real")
    resize_images(train_fake_dir, "/Users/guanz/Documents/cs229/project/real_fake_faces_140k/real_vs_fake/resized/train_sampled/fake")

    resize_images(val_real_dir, "/Users/guanz/Documents/cs229/project/real_fake_faces_140k/real_vs_fake/resized/validation_sampled/real")
    resize_images(val_fake_dir, "/Users/guanz/Documents/cs229/project/real_fake_faces_140k/real_vs_fake/resized/validation_sampled/fake")

    end_time = time.time()
    print(f"Processing completed in {end_time - start_time:.2f} seconds")
