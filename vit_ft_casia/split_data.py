# split_data.py

import os
import random
from pathlib import Path
import shutil
from collections import defaultdict

# Set seed for reproducibility
random.seed(42)

# Paths
base_path = Path("/Users/guanz/Documents/cs229/project/CASIA2.0_revised")
au_path = base_path / "Au"
tp_path = base_path / "Tp"
output_dir = base_path / "splits"

# Read file lists
with open("/Users/guanz/Documents/cs229/project/CASIA2.0_revised/au_list.txt", "r") as f:
    au_files = [line.strip() for line in f.readlines()]

with open("/Users/guanz/Documents/cs229/project/CASIA2.0_revised/tp_list.txt", "r") as f:
    tp_files = [line.strip() for line in f.readlines()]

# Helper function to categorize files
def categorize_au_files(files):
    categories = defaultdict(list)
    for file in files:
        category = file.split("_")[1]  # Extract category (e.g., ani, arc)
        categories[category].append(file)
    return categories

def categorize_tp_files(files):
    categories = defaultdict(lambda: defaultdict(list))
    for file in files:
        tamper_type = file.split("_")[2]  # 'D' or 'S'
        source_category = file.split("_")[5][:3]  # Source category (e.g., ani, arc)
        target_category = file.split("_")[6][:3]  # Target category (e.g., ani, arc)
        categories[tamper_type][(source_category, target_category)].append(file)
    return categories

# Categorize files
au_categories = categorize_au_files(au_files)
tp_categories = categorize_tp_files(tp_files)

# Split data into train/dev/test
def split_category(category_data, train_ratio=0.7, dev_ratio=0.15):
    train, dev, test = [], [], []
    for files in category_data.values():
        random.shuffle(files)
        train_end = int(len(files) * train_ratio)
        dev_end = train_end + int(len(files) * dev_ratio)
        train.extend(files[:train_end])
        dev.extend(files[train_end:dev_end])
        test.extend(files[dev_end:])
    return train, dev, test

# Split each category
au_train, au_dev, au_test = split_category(au_categories)
tp_train, tp_dev, tp_test = [], [], []
for tamper_type, subcategories in tp_categories.items():
    train, dev, test = split_category(subcategories)
    tp_train.extend(train)
    tp_dev.extend(dev)
    tp_test.extend(test)

# Prepare output directories
for split in ["train", "dev", "test"]:
    for label in ["Au", "Tp"]:
        (output_dir / split / label).mkdir(parents=True, exist_ok=True)

# Helper function to copy files
def copy_files(file_list, source_dir, dest_dir):
    for file in file_list:
        src_path = source_dir / file
        dest_path = dest_dir / file
        if src_path.exists():
            shutil.copy(src_path, dest_path)
        else:
            print(f"File {file} not found in {source_dir}")

# Copy files to respective directories
copy_files(au_train, au_path, output_dir / "train" / "Au")
copy_files(au_dev, au_path, output_dir / "dev" / "Au")
copy_files(au_test, au_path, output_dir / "test" / "Au")

copy_files(tp_train, tp_path, output_dir / "train" / "Tp")
copy_files(tp_dev, tp_path, output_dir / "dev" / "Tp")
copy_files(tp_test, tp_path, output_dir / "test" / "Tp")

print("Dataset successfully split with balanced categories!")
