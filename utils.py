import json
import math
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────

class BedDataset(Dataset):
    def __init__(self, root, split, transform=None, augment=False):
        """
        root: 数据集根目录，含 train/val 子目录和 train_labels.json/val_labels.json
        augment: 是否启用水平翻转（训练时开启，翻转时同步修正 yaw）
        """
        self.img_dir = Path(root) / split
        self.transform = transform
        self.augment = augment
        with open(Path(root) / f"{split}_labels.json") as f:
            labels = json.load(f)
        self.items = list(labels.items())  # [(img_id, yaw_deg), ...]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_id, yaw_deg = self.items[idx]
        img = Image.open(self.img_dir / f"{img_id}.png").convert("RGB")

        if self.augment and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            yaw_deg = (180.0 - yaw_deg) % 360.0

        if self.transform:
            img = self.transform(img)

        yaw_rad = math.radians(yaw_deg)
        gt = torch.tensor([math.sin(yaw_rad), math.cos(yaw_rad)], dtype=torch.float32)
        return img, gt


# ──────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────

def build_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomCrop(224, padding=16, fill=128),  # padding 后 crop，椅子始终完整
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


# ──────────────────────────────────────────────
# Loss & Metrics
# ──────────────────────────────────────────────

def angular_loss(pred, gt):
    """
    pred, gt: (B, 2) 单位向量 (sin, cos)
    返回 batch 平均角度误差（度），自然 clip 在 [0, 180]
    """
    cos_diff = (pred * gt).sum(dim=1).clamp(-1 + 1e-7, 1 - 1e-7)
    err_deg = torch.acos(cos_diff) * (180.0 / math.pi)
    return err_deg.mean()


@torch.no_grad()
def evaluate(model, loader, device):
    """返回 dict: mae, acc@5°, acc@10°, acc@15°, acc@30°"""
    model.eval()
    errors = []
    for imgs, gts in loader:
        imgs, gts = imgs.to(device), gts.to(device)
        preds = model(imgs)
        cos_diff = (preds * gts).sum(dim=1).clamp(-1 + 1e-7, 1 - 1e-7)
        err_deg = torch.acos(cos_diff) * (180.0 / math.pi)
        errors.append(err_deg.cpu())
    errors = torch.cat(errors)
    return {
        "mae":   errors.mean().item(),
        "acc5":  (errors < 5).float().mean().item()  * 100,
        "acc10": (errors < 10).float().mean().item() * 100,
        "acc15": (errors < 15).float().mean().item() * 100,
        "acc30": (errors < 30).float().mean().item() * 100,
    }


def pred_to_yaw(pred_vec):
    """(sin, cos) → yaw 角度（度），单个样本或 batch 均可"""
    if pred_vec.dim() == 1:
        sin_y, cos_y = pred_vec[0].item(), pred_vec[1].item()
        return math.degrees(math.atan2(sin_y, cos_y)) % 360.0
    yaw = torch.atan2(pred_vec[:, 0], pred_vec[:, 1]) * (180.0 / math.pi)
    return yaw % 360.0
