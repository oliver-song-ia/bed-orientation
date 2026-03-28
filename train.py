"""
训练床朝向 Circle Regression 模型

用法:
    python train.py [--epochs 60] [--bs 64] [--lr 1e-4] [--warmup 3]
    python train.py --resume  # 从上次 best_model.pth 继续训练

TensorBoard:
    tensorboard --logdir /home/tom-wang/Documents/bed-orientation/runs
"""

import argparse
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from net import YawRegressor
from utils import BedDataset, build_transforms, angular_loss, evaluate

DATA_DEFAULT   = "/home/tom-wang/Documents/data/bed-orientation"
OUTPUT_DEFAULT = "/home/tom-wang/Documents/bed-orientation"


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir   = Path(args.output)
    save_path = out_dir / "best_model.pth"

    train_tf, val_tf = build_transforms()
    train_ds = BedDataset(args.data, "train", train_tf, augment=True)
    val_ds   = BedDataset(args.data, "val",   val_tf,   augment=False)
    train_dl = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                          num_workers=4, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.bs, shuffle=False,
                          num_workers=4, pin_memory=True)
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # ── 模型 ──
    model = YawRegressor(backbone=args.backbone).to(device)
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": args.lr * 0.1},
        {"params": model.head.parameters(),     "lr": args.lr},
    ], weight_decay=1e-4)
    # T_max 对应解冻后的有效训练 epoch 数，让 LR 在结束时才降到最低
    effective_epochs = args.epochs - args.warmup
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=effective_epochs, eta_min=1e-6
    )

    start_epoch = 1
    best_mae    = float("inf")

    # ── Resume ──
    if args.resume and save_path.exists():
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_mae    = ckpt.get("best_mae", float("inf"))
        # 恢复 scheduler 到对应步数
        resume_steps = max(0, start_epoch - 1 - args.warmup)
        for _ in range(resume_steps):
            scheduler.step()
        print(f"Resume from epoch {start_epoch}  best_mae={best_mae:.2f}°")
    else:
        print(f"Device: {device}  Backbone: {args.backbone}")

    run_name = f"{args.backbone}_{datetime.now().strftime('%m%d_%H%M')}"
    log_dir  = out_dir / "runs" / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    writer   = SummaryWriter(log_dir=str(log_dir))
    print(f"TensorBoard log: {log_dir}")

    for epoch in range(start_epoch, args.epochs + 1):
        # warmup: 冻结 backbone
        if epoch <= args.warmup:
            for p in model.backbone.parameters():
                p.requires_grad_(False)
        elif epoch == args.warmup + 1:
            for p in model.backbone.parameters():
                p.requires_grad_(True)
            print(f"Epoch {epoch}: 解冻 backbone")

        # ── train ──
        model.train()
        total_loss = 0.0
        for imgs, gts in train_dl:
            imgs, gts = imgs.to(device), gts.to(device)
            loss = angular_loss(model(imgs), gts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(imgs)

        if epoch > args.warmup:
            scheduler.step()

        train_loss = total_loss / len(train_ds)

        # ── val ──
        metrics  = evaluate(model, val_dl, device)
        val_loss = metrics["mae"]

        # ── TensorBoard ──
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalar("Metrics/MAE",    metrics["mae"],   epoch)
        writer.add_scalar("Metrics/acc@5",  metrics["acc5"],  epoch)
        writer.add_scalar("Metrics/acc@10", metrics["acc10"], epoch)
        writer.add_scalar("Metrics/acc@15", metrics["acc15"], epoch)
        writer.add_scalar("Metrics/acc@30", metrics["acc30"], epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

        # ── checkpoint ──
        mark = " ★" if val_loss < best_mae else ""
        if val_loss < best_mae:
            best_mae = val_loss
            torch.save({"backbone":   args.backbone,
                        "state_dict": model.state_dict(),
                        "epoch":      epoch,
                        "best_mae":   best_mae}, save_path)

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={train_loss:.2f}°  val={val_loss:.2f}°  "
              f"acc@5°={metrics['acc5']:.1f}%  acc@10°={metrics['acc10']:.1f}%"
              f"{mark}")

    writer.close()
    print(f"\n最优 val MAE: {best_mae:.2f}°  已保存至 {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",     default=DATA_DEFAULT)
    parser.add_argument("--output",   default=OUTPUT_DEFAULT)
    parser.add_argument("--backbone", default="resnet50", choices=["resnet50", "resnet18"])
    parser.add_argument("--epochs",   type=int,   default=60)
    parser.add_argument("--bs",       type=int,   default=64)
    parser.add_argument("--lr",       type=float, default=1e-4)
    parser.add_argument("--warmup",   type=int,   default=3,
                        help="前 N 个 epoch 冻结 backbone 只训练 head")
    parser.add_argument("--resume",   action="store_true",
                        help="从 best_model.pth 继续训练")
    args = parser.parse_args()
    train(args)
