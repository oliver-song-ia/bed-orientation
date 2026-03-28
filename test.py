"""
测试 / 推理脚本

用法:
    # 在 val set 上评估
    python test.py --ckpt /path/to/best_model.pth

    # 对单张图片推理
    python test.py --ckpt /path/to/best_model.pth --image /path/to/chair.png

    # 可视化 val set 预测结果（随机 N 张）
    python test.py --ckpt /path/to/best_model.pth --vis --n 16
"""

import argparse
import math
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader

from net import YawRegressor
from utils import BedDataset, build_transforms, evaluate, pred_to_yaw

DATA_DEFAULT = "/home/tom-wang/Documents/data/bed-orientation"
CKPT_DEFAULT = "/home/tom-wang/Documents/bed-orientation/best_model.pth"


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    backbone = ckpt.get("backbone", "resnet50")
    model = YawRegressor(backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def infer_single(model, img_path, device):
    """对单张图片推理，返回预测 yaw 角度（度）"""
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x)[0]
    return pred_to_yaw(pred)


def draw_yaw_arrow(ax, yaw_deg, color, label=None):
    """在 ax 中心绘制 yaw 朝向箭头（0°=左，CW+）"""
    lims = ax.get_xlim(), ax.get_ylim()
    cx = (lims[0][0] + lims[0][1]) / 2
    cy = (lims[1][0] + lims[1][1]) / 2
    r = min(abs(lims[0][1] - lims[0][0]), abs(lims[1][1] - lims[1][0])) * 0.3
    display_rad = math.radians((180 - yaw_deg) % 360)
    dx = r * math.cos(display_rad)
    dy = -r * math.sin(display_rad)  # image y 轴向下
    ax.annotate("", xy=(cx + dx, cy + dy), xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2.5))
    if label:
        ax.set_xlabel(label, fontsize=8)


def visualize_val(model, args, device):
    import random
    _, val_tf = build_transforms()
    val_ds = BedDataset(args.data, "val", val_tf, augment=False)
    indices = random.sample(range(len(val_ds)), min(args.n, len(val_ds)))

    cols = 4
    rows = (len(indices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.8))
    axes = np.array(axes).reshape(-1)

    for ax, idx in zip(axes[:len(indices)], indices):
        img_tensor, gt_vec = val_ds[idx]
        gt_yaw = pred_to_yaw(gt_vec)

        with torch.no_grad():
            pred_vec = model(img_tensor.unsqueeze(0).to(device))[0].cpu()
        pred_yaw = pred_to_yaw(pred_vec)

        cos_diff = (pred_vec * gt_vec).sum().clamp(-1 + 1e-7, 1 - 1e-7)
        err = math.degrees(math.acos(cos_diff.item()))

        # 还原图片显示
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_show = (img_tensor * std + mean).permute(1, 2, 0).clamp(0, 1).numpy()

        ax.imshow(img_show)
        ax.axis("off")
        draw_yaw_arrow(ax, gt_yaw,   color="green", label=None)
        draw_yaw_arrow(ax, pred_yaw, color="red",   label=None)
        ax.set_title(f"GT={gt_yaw:.0f}°  Pred={pred_yaw:.0f}°\nerr={err:.1f}°", fontsize=7)

    for ax in axes[len(indices):]:
        ax.axis("off")

    fig.legend(handles=[
        plt.Line2D([0], [0], color="green", lw=2, label="GT"),
        plt.Line2D([0], [0], color="red",   lw=2, label="Pred"),
    ], loc="lower center", ncol=2, fontsize=10)

    plt.suptitle("Val predictions", fontsize=12)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",    default=CKPT_DEFAULT)
    parser.add_argument("--data",    default=DATA_DEFAULT)
    parser.add_argument("--image",   default=None, help="单张图片路径")
    parser.add_argument("--vis",     action="store_true", help="可视化 val set 预测")
    parser.add_argument("--n",       type=int, default=16, help="可视化样本数")
    parser.add_argument("--bs",      type=int, default=64)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)
    print(f"模型已加载: {args.ckpt}")

    if args.image:
        yaw = infer_single(model, args.image, device)
        print(f"预测 yaw: {yaw:.1f}°")
        return

    # 在 val set 上评估
    _, val_tf = build_transforms()
    val_ds = BedDataset(args.data, "val", val_tf, augment=False)
    val_dl = DataLoader(val_ds, batch_size=args.bs, num_workers=4, pin_memory=True)
    m = evaluate(model, val_dl, device)
    print(f"Val MAE={m['mae']:.2f}°  "
          f"acc@5°={m['acc5']:.1f}%  acc@10°={m['acc10']:.1f}%  "
          f"acc@15°={m['acc15']:.1f}%  acc@30°={m['acc30']:.1f}%")

    if args.vis:
        visualize_val(model, args, device)


if __name__ == "__main__":
    main()
