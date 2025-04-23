import yaml
import torch
import numpy as np
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import pandas as pd
import os
import json 
from PIL import Image

def load_config(path="../config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def plot_metrics(metrics: dict, name: str, ylabel: str, save_dir: str):
    plt.figure()
    for label, values in metrics.items():
        plt.plot(values, label=label)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{name.replace(' ', '_').lower()}.png"))
    plt.close()

def save_metrics_json_csv(history: dict, json_path: str, csv_path: str):
    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(history, f, indent=4)

    # Prepare for CSV
    max_len = max(len(v) if isinstance(v, list) else 0 for v in history.values())
    data = {}

    for key, value in history.items():
        if isinstance(value, list):
            if all(isinstance(v, list) for v in value):  # per-class metrics
                for i, sublist in enumerate(value):
                    label = f"{key}_class_{i}"
                    data[label] = sublist + [None] * (max_len - len(sublist))
            else:
                data[key] = value + [None] * (max_len - len(value))

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index_label="Epoch")

def overlay_mask(image_tensor, mask):

    image = TF.to_pil_image(image_tensor)
    image_np = np.array(image)
    overlay = image_np.copy()

    # Define class color map
    class_colors = {
        1: [255, 165, 0],     # lane - orange
        2: [100, 255, 100],     # lane_1 - cyan
        3: [255, 105, 180],   # lane_3 - pink
        4: [255, 192, 203],   # lane_2 - pink variant
        5: [0, 0, 139],       # my_lane - dark blue
        6: [0, 255, 255],     # other_lane - cyan
        7: [128, 0, 128],     # road - purple
    }

    class_names = {
        1: "lane",
        2: "lane_1",
        3: "lane_3",
        4: "lane_2",
        5: "my_lane",
        6: "other_lane",
        7: "road"
    }

    # Log what classes are present
    present_classes = np.unique(mask)
    print("Detected classes in image:")
    for class_id in present_classes:
        if class_id in class_names:
            print(f"  - {class_names[class_id]} (Class ID: {class_id})")

    # Apply color overlay
    for class_id, color in class_colors.items():
        overlay[mask == class_id] = color

    return TF.to_tensor(overlay)
