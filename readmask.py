import os
import cv2
import numpy as np
from glob import glob

# Define directories
train_mask_dir = "data/processed/train/masks"
val_mask_dir = "data/processed/val/masks"

def process_mask(mask_path):
    """Loads a mask, scales pixel values, and saves it back."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Could not read {mask_path}")
        return
    
    # Scale up pixel values for better visibility
    mask_visual = (mask * 255).astype(np.uint8)
    
    # Save the modified mask
    cv2.imwrite(mask_path, mask_visual)
    print(f"Processed and saved: {mask_path}")

# Process all masks in train and val directories
for mask_dir in [train_mask_dir, val_mask_dir]:
    mask_paths = sorted(glob(os.path.join(mask_dir, "*.png")))
    if not mask_paths:
        print(f"No masks found in {mask_dir}")
    else:
        for mask_path in mask_paths:
            process_mask(mask_path)

print("Mask processing complete!")