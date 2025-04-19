import yaml
import torch
import numpy as np
import torchvision.transforms.functional as TF


def load_config(path="../config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def save_model(model, path):
    torch.save(model.state_dict(), path)


def overlay_mask(image_tensor, mask):
    import torchvision.transforms.functional as TF
    import numpy as np
    from PIL import Image

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
