import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import LaneDataset
from model import UNet
from utils import load_config, save_model, plot_metrics, save_metrics_json_csv
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
import glob


def compute_iou(preds, labels, num_classes):
    ious = []
    preds = preds.view(-1)
    labels = labels.view(-1)
    for cls in range(num_classes):
        pred_inds = preds == cls
        label_inds = labels == cls
        intersection = (pred_inds & label_inds).sum().item()
        union = (pred_inds | label_inds).sum().item()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return ious


def compute_class_metrics(preds, labels, num_classes):
    preds = preds.view(-1).cpu().numpy()
    labels = labels.view(-1).cpu().numpy()
    precision = precision_score(labels, preds, average=None, labels=list(range(num_classes)), zero_division=0)
    recall = recall_score(labels, preds, average=None, labels=list(range(num_classes)), zero_division=0)
    f1 = f1_score(labels, preds, average=None, labels=list(range(num_classes)), zero_division=0)
    dice = 2 * (precision * recall) / (precision + recall + 1e-7)
    return precision, recall, f1, dice


def train():
    config = load_config()
    os.makedirs("graphs", exist_ok=True)

    train_images = sorted(glob.glob(os.path.join(config['data']['train_images'], "*")))
    train_masks = sorted(glob.glob(os.path.join(config['data']['train_masks'], "*")))
    val_images = sorted(glob.glob(os.path.join(config['data']['val_images'], "*")))
    val_masks = sorted(glob.glob(os.path.join(config['data']['val_masks'], "*")))

    print(f"Train images: {len(train_images)}")
    print(f"Train masks:  {len(train_masks)}")
    print(f"Val images:   {len(val_images)}")
    print(f"Val masks:    {len(val_masks)}")

    train_dataset = LaneDataset(config['data']['train_images'], config['data']['train_masks'], config['training']['image_size'])
    val_dataset = LaneDataset(config['data']['val_images'], config['data']['val_masks'], config['training']['image_size'])

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model = UNet(num_classes=config['training']['num_classes']).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "miou": [],
        "precision": [[] for _ in range(config['training']['num_classes'])],
        "recall": [[] for _ in range(config['training']['num_classes'])],
        "f1": [[] for _ in range(config['training']['num_classes'])],
        "dice": [[] for _ in range(config['training']['num_classes'])],
        "per_class_iou": [[] for _ in range(config['training']['num_classes'])],
    }

    for epoch in range(config['training']['epochs']):
        model.train()
        running_loss = 0
        correct_pixels = 0
        total_pixels = 0

        print(f"\nEpoch {epoch+1}/{config['training']['epochs']}")
        for images, masks in tqdm(train_loader, desc="Training", leave=False):
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

        train_loss = running_loss / len(train_loader)
        train_acc = correct_pixels / total_pixels
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        total_ious = []
        with torch.no_grad():
            for images, masks in tqdm(val_loader, desc="Validating", leave=False):
                images, masks = images.cuda(), masks.cuda()
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == masks).sum().item()
                val_total += torch.numel(masks)

                ious = compute_iou(preds, masks, config['training']['num_classes'])
                total_ious.append(ious)

                all_preds.append(preds.view(-1))
                all_labels.append(masks.view(-1))

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        if len(total_ious) == 0:
            avg_ious = [0.0 for _ in range(config['training']['num_classes'])]
            miou = 0.0
        else:
            avg_ious = np.nanmean(total_ious, axis=0)
            miou = np.nanmean(avg_ious)


        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        precision, recall, f1, dice = compute_class_metrics(all_preds, all_labels, config['training']['num_classes'])

        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["miou"].append(miou)

        for i in range(config['training']['num_classes']):
            history["per_class_iou"][i].append(avg_ious[i])
            history["precision"][i].append(precision[i])
            history["recall"][i].append(recall[i])
            history["f1"][i].append(f1[i])
            history["dice"][i].append(dice[i])

        print(f"Train Loss: {train_loss:.4f} | Pixel Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Pixel Acc: {val_acc:.4f} | mIoU: {miou:.4f}")

    print("\nTraining complete. Saving model...")
    save_model(model, config['training']['save_path'])
    print(f"Model saved to {config['training']['save_path']}")

    # === Plotting ===
    plot_metrics({"Train Loss": history["train_loss"], "Val Loss": history["val_loss"]}, "Loss Curve", "Loss", "graphs")
    plot_metrics({"Train Accuracy": history["train_acc"], "Val Accuracy": history["val_acc"]}, "Pixel Accuracy", "Accuracy", "graphs")
    plot_metrics({"mIoU": history["miou"]}, "Mean IoU", "mIoU", "graphs")

    for i in range(config['training']['num_classes']):
        plot_metrics({f"Class {i} IoU": history["per_class_iou"][i]}, f"IoU for Class {i}", "IoU", "graphs")
        plot_metrics({f"Class {i} Precision": history["precision"][i]}, f"Precision for Class {i}", "Precision", "graphs")
        plot_metrics({f"Class {i} Recall": history["recall"][i]}, f"Recall for Class {i}", "Recall", "graphs")
        plot_metrics({f"Class {i} F1": history["f1"][i]}, f"F1-score for Class {i}", "F1", "graphs")
        plot_metrics({f"Class {i} Dice": history["dice"][i]}, f"Dice Coefficient for Class {i}", "Dice", "graphs")

    save_metrics_json_csv(history, "graphs/metrics.json", "graphs/metrics.csv")


if __name__ == "__main__":
    train()