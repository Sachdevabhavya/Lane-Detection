import torch
from model import UNet
from utils import load_config, overlay_mask
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import os

def predict_single_image():
    config = load_config()
    
    image_path = "../test/1478019952686311006.jpg"

    # Load model
    model = UNet(num_classes=config['training']['num_classes']).cuda()
    model.load_state_dict(torch.load(config['training']['save_path']))
    model.eval()

    # Image transform
    transform = T.Compose([
        T.Resize(config['training']['image_size']),
        T.ToTensor(),
    ])

    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).cuda()  # [1, 3, H, W]

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

    # Overlay and save result
    overlay = overlay_mask(input_tensor[0].cpu(), pred_mask)
    os.makedirs("../outputs", exist_ok=True)
    TF.to_pil_image(overlay).save("../outputs/live_result_2.png")
    print("Result generated")

if __name__ == "__main__":
    predict_single_image()
