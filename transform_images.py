# transform_images.py

from PIL import Image
import os

def resize_images(source_dir, target_dir, size=(224, 224)):
    os.makedirs(target_dir, exist_ok=True)
    for img_name in os.listdir(source_dir):
        img_path = os.path.join(source_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img = img.resize(size)
        img.save(os.path.join(target_dir, img_name))

if __name__ == "__main__":
    # Paths for original and target resized images
    train_real_dir = "/Users/guanz/Documents/cs229/project/CIFake/train/REAL"
    train_fake_dir = "/Users/guanz/Documents/cs229/project/CIFake/train/FAKE"
    test_real_dir = "/Users/guanz/Documents/cs229/project/CIFake/test/REAL"
    test_fake_dir = "/Users/guanz/Documents/cs229/project/CIFake/test/FAKE"
    
    resize_images(train_real_dir, "/Users/guanz/Documents/cs229/project/CIFake/resized/train/REAL")
    resize_images(train_fake_dir, "/Users/guanz/Documents/cs229/project/CIFake/resized/train/FAKE")
    resize_images(test_real_dir, "/Users/guanz/Documents/cs229/project/CIFake/resized/test/REAL")
    resize_images(test_fake_dir, "/Users/guanz/Documents/cs229/project/CIFake/resized/test/FAKE")
