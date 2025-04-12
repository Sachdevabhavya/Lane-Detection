### src/train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import LaneDataset
from model import UNet
from utils import load_config, save_model
import os

def train():
    config = load_config()
    print("Configuration loaded.")
    print("Loading datasets...")
    train_dataset = LaneDataset(config['data']['train_images'], config['data']['train_masks'], config['training']['image_size'])
    val_dataset = LaneDataset(config['data']['val_images'], config['data']['val_masks'], config['training']['image_size'])

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = UNet(num_classes=config['training']['num_classes']).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    for epoch in range(config['training']['epochs']):
        model.train()
        running_loss = 0.0
        correct_pixels = 0
        total_pixels = 0

        for images, masks in train_loader:
            images, masks = images.cuda(), masks.cuda()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct_pixels += (preds == masks).sum().item()
            total_pixels += torch.numel(masks)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct_pixels / total_pixels

        print(f"Epoch {epoch+1}/{config['training']['epochs']} - Loss: {epoch_loss:.4f} - Pixel Accuracy: {epoch_acc:.4f}")

    print("Training complete. Saving model...")
    save_model(model, config['training']['save_path'])
    print(f"Model saved to {config['training']['save_path']}")

if __name__ == "__main__":
    train()
