import os
import numpy as np
import tensorflow as tf
from glob import glob

# Define constants
IMG_HEIGHT, IMG_WIDTH = 512, 512
BATCH_SIZE = 8

def parse_data(image_path, mask_path):
    """
    Reads, resizes, and normalizes image and mask.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH], method='nearest')
    mask = tf.cast(mask, tf.float32) / 255.0  

    return image, mask

def load_data(train_image_dir, train_mask_dir, val_image_dir, val_mask_dir):
    """
    Loads training and validation datasets separately.
    """
    train_image_paths = sorted(glob(os.path.join(train_image_dir, "*.png")))
    train_mask_paths = sorted(glob(os.path.join(train_mask_dir, "*.png")))
    val_image_paths = sorted(glob(os.path.join(val_image_dir, "*.png")))
    val_mask_paths = sorted(glob(os.path.join(val_mask_dir, "*.png")))

    print(f"Training images found: {len(train_image_paths)}")
    print(f"Training masks found: {len(train_mask_paths)}")
    print(f"Validation images found: {len(val_image_paths)}")
    print(f"Validation masks found: {len(val_mask_paths)}")

    # Create training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_mask_paths))
    train_dataset = train_dataset.map(parse_data, num_parallel_calls=tf.data.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=len(train_image_paths), seed=42)
    train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Create validation dataset
    val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_mask_paths))
    val_dataset = val_dataset.map(parse_data, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, val_dataset
