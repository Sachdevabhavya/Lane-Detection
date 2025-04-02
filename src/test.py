import os
import cv2
import numpy as np
import tensorflow as tf
from model import dice_coef
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from glob import glob

# Constants
IMG_HEIGHT, IMG_WIDTH = 512, 512
THRESHOLD = 0.05
TEST_IMAGE_DIR = "../test"

# Load Model
model = load_model("../models/final_unet_road_segmentation_model_v0.0.1.h5", custom_objects={'dice_coef': dice_coef})

def preprocess_image(image):
    return cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT)).astype("float32") / 255.0

def predict_mask(image):
    input_img = np.expand_dims(preprocess_image(image), axis=0)
    pred = model.predict(input_img)[0, :, :, 0]
    
    # Debugging: Print min/max values in prediction
    print(f"Prediction Min: {np.min(pred)}, Max: {np.max(pred)}")

    above_threshold = np.sum(pred > THRESHOLD)
    print(f"Pixels above threshold: {above_threshold}")

    return (pred > THRESHOLD).astype(np.uint8) * 255


def overlay_mask(image, mask):
    """
    Overlays the segmentation mask on the original image and marks lane lines in color.
    """
    color_mask = np.zeros_like(image)
    color_mask[mask == 255] = (0, 255, 0)  # Green for segmentation mask
    
    # Convert mask to edges
    edges = cv2.Canny(mask, 50, 150)
    
    # Find lane contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw lane lines on the image
    lane_overlay = image.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Filter small noise
            cv2.polylines(lane_overlay, [contour], isClosed=False, color=(0, 0, 255), thickness=2)  # Red lane lines
    
    return cv2.addWeighted(lane_overlay, 0.7, color_mask, 0.3, 0)

def test_on_image(image_path, save_dir="../results"):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    mask = predict_mask(image)
    result = overlay_mask(cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT)), mask)
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save result image
    save_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(save_path, result)
    print(f"Saved result image at: {save_path}")

    # Display results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1); plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)); plt.title("Original Image")
    plt.subplot(1, 3, 2); plt.imshow(mask, cmap="gray"); plt.title("Predicted Mask")
    plt.subplot(1, 3, 3); plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)); plt.title("Overlay Result with Lane Markings")
    plt.show()


if __name__ == "__main__":
    image_paths = sorted(glob(os.path.join(TEST_IMAGE_DIR, "Image_0.png")))

    if not image_paths:
        print(f"No test images found in {TEST_IMAGE_DIR}")
    else:
        for image_path in image_paths:
            print(f"Processing: {image_path}")
            test_on_image(image_path)
