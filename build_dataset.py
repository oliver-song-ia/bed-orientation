"""
从三个来源构建训练数据集，每个来源独立按 9:1 划分 train/val：
  - bed_selected        : 全部使用（手选真实数据，COCO）
  - bed_office_selected : 全部使用（手选真实数据，办公室录制）
  - output              : 全部使用（合成数据）

保存至 /home/tom-wang/Documents/data/bed-orientation
"""

import json
import os
import random
import shutil
from pathlib import Path

ASSET_DIR   = Path(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_ROOT = Path("/home/tom-wang/Documents/data/bed-orientation")
TRAIN_RATIO = 0.9
SEED        = 42

BED_SELECTED_DIRS = [
    Path("/home/tom-wang/Documents/data/furniture_orientation/bed_selected"),
    Path("/home/tom-wang/Documents/data/furniture_orientation/bed_office_selected"),
]


def collect_bed_selected():
    """bed_selected / bed_office_selected: labels.json 是 {id: {image, yaw, ...}}"""
    samples = []
    for d in BED_SELECTED_DIRS:
        labels_file = d / "labels.json"
        img_dir     = d / "images"
        if not labels_file.exists():
            print(f"  {d.name}: no labels.json, skipping")
            continue
        with open(labels_file) as f:
            labels = json.load(f)
        hit = miss = 0
        for ann_id, ann in labels.items():
            img_path = img_dir / ann["image"]
            if img_path.exists():
                samples.append((img_path, float(ann["yaw"])))
                hit += 1
            else:
                miss += 1
        print(f"  {d.name}: {hit} available, {miss} missing")
    return samples


def collect_synthetic():
    """output: labels_0_2000.json 是 [{image, rotation_yaw_degrees}, ...]"""
    labels_file = ASSET_DIR / "output" / "labels_0_2000.json"
    images_dir  = ASSET_DIR / "output" / "bed_images"

    with open(labels_file) as f:
        labels = json.load(f)

    samples = []
    miss = 0
    for entry in labels:
        img_path = images_dir / entry["image"]
        if img_path.exists():
            samples.append((img_path, float(entry["rotation_yaw_degrees"])))
        else:
            miss += 1

    print(f"  {'output':<25}: {len(samples)} available, {miss} missing")
    return samples


def split_source(samples, ratio=TRAIN_RATIO):
    """对单个来源做 9:1 划分"""
    random.shuffle(samples)
    n_train = int(len(samples) * ratio)
    return samples[:n_train], samples[n_train:]


def save_split(samples, split, offset=0):
    """将样本写入 OUTPUT_ROOT/split，返回写入数量"""
    out_dir = OUTPUT_ROOT / split
    out_dir.mkdir(parents=True, exist_ok=True)
    label_file = OUTPUT_ROOT / f"{split}_labels.json"

    if label_file.exists():
        with open(label_file) as f:
            labels = json.load(f)
    else:
        labels = {}

    for i, (img_path, yaw) in enumerate(samples):
        new_id   = f"bed_{offset + i:05d}"
        new_name = f"{new_id}.png"
        shutil.copy2(img_path, out_dir / new_name)
        labels[new_id] = yaw

    with open(label_file, "w") as f:
        json.dump(labels, f, indent=2)

    return len(samples)


def main():
    random.seed(SEED)

    print("Collecting samples...")
    sel_samples = collect_bed_selected()
    syn_samples = collect_synthetic()

    total = len(sel_samples) + len(syn_samples)
    print(f"\nTotal: bed_selected={len(sel_samples)}  output={len(syn_samples)}  total={total}")

    # 每个来源独立 9:1 划分
    sel_train, sel_val = split_source(sel_samples)
    syn_train, syn_val = split_source(syn_samples)

    n_train = len(sel_train) + len(syn_train)
    n_val   = len(sel_val)   + len(syn_val)
    print(f"Train: {n_train}  Val: {n_val}")
    print(f"  bed_selected    train={len(sel_train)}  val={len(sel_val)}")
    print(f"  output          train={len(syn_train)}  val={len(syn_val)}")

    # 清空旧数据集
    for split in ("train", "val"):
        split_dir = OUTPUT_ROOT / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True)
        label_file = OUTPUT_ROOT / f"{split}_labels.json"
        if label_file.exists():
            label_file.unlink()

    # 打乱后写入
    train_all = sel_train + syn_train
    val_all   = sel_val   + syn_val
    random.shuffle(train_all)
    random.shuffle(val_all)

    print("\nSaving train...")
    save_split(train_all, "train", offset=0)
    print("Saving val...")
    save_split(val_all, "val", offset=0)

    print(f"\nDone! Saved to {OUTPUT_ROOT}")
    print(f"  train: {len(train_all)} samples")
    print(f"  val:   {len(val_all)} samples")


if __name__ == "__main__":
    main()
