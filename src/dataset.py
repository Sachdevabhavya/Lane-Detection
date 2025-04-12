import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class LaneDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_size):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_size = tuple(image_size)
        self.images = sorted(os.listdir(images_dir))
        self.masks = sorted(os.listdir(masks_dir))
        self.image_transform = T.Compose([
            T.Resize(self.image_size),
            T.ToTensor(),
        ])
        self.mask_transform = T.Compose([
            T.Resize(self.image_size, interpolation=Image.NEAREST),
            T.PILToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.masks[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        image = self.image_transform(image)
        mask = self.mask_transform(mask)[0].long()  # Use red channel as class ID
        return image, mask
