import os
import shutil
import random

# Define paths
dataset_path = "images"  # Folder containing all images
train_path = "images/train"  # Destination for training images
val_path = "images/val"  # Destination for validation images

# Create directories if they don't exist
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)

# Get all image files
all_images = [f for f in os.listdir(dataset_path) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Shuffle images randomly
random.shuffle(all_images)

# Split into training and validation sets (80% train, 20% val)
split_idx = int(len(all_images) * 0.8)
train_images = all_images[:split_idx]
val_images = all_images[split_idx:]

# Move images to respective folders
for img in train_images:
    shutil.move(os.path.join(dataset_path, img), os.path.join(train_path, img))

for img in val_images:
    shutil.move(os.path.join(dataset_path, img), os.path.join(val_path, img))

print(f"Moved {len(train_images)} images to {train_path}")
print(f"Moved {len(val_images)} images to {val_path}")
