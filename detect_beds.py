#!/usr/bin/env python3
"""
使用自定义YOLO模型检测合成图中的床并保存bbox裁剪结果
每张图有且仅有一个床，取置信度最高的检测结果

用法:
    python detect_beds.py
    python detect_beds.py --no-viz
"""

import os
import cv2
import argparse
import warnings
from pathlib import Path

import numpy as np
from ultralytics import YOLO

warnings.filterwarnings("ignore")

ASSET_DIR  = Path(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = "/home/tom-wang/Documents/custom_segmentation/runs/segment/runs/segment/yolo26m-seg-custom-v2/weights/best.pt"
BED_CLASS_ID = 2  # {0: person, 1: chair, 2: bed, 3: toilet, 4: wheelchair}


def detect_and_crop(model, image, conf_threshold=0.01, iou_threshold=0.45):
    """检测图中的床，返回置信度最高的 bbox 裁剪，找不到则返回 None"""
    results = model.predict(
        image,
        classes=[BED_CLASS_ID],
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
        max_det=10,
    )
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return None, None

    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    best  = int(np.argmax(confs))
    x1, y1, x2, y2 = boxes[best].astype(int)
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return image[y1:y2, x1:x2], (x1, y1, x2, y2, float(confs[best]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir",  default=str(ASSET_DIR / "output" / "images"))
    parser.add_argument("--output-dir", default=None,
                        help="输出根目录（默认为 input-dir 的上一级）")
    parser.add_argument("--no-viz",     action="store_true", help="不保存可视化图片")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir.parent
    crops_dir  = output_dir / "bed_images"
    crops_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_viz:
        viz_dir = output_dir / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

    print(f"加载模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print("模型加载完成\n")

    image_files = sorted(input_dir.glob("*.png")) + sorted(input_dir.glob("*.jpg"))
    if not image_files:
        print(f"错误: {input_dir} 中没有图片")
        return

    print(f"共 {len(image_files)} 张图片，输出至: {crops_dir}\n")

    n_found = n_miss = 0
    for i, img_path in enumerate(image_files):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[{i+1}/{len(image_files)}] 无法读取: {img_path.name}")
            continue

        crop, det = detect_and_crop(model, image)

        if crop is None or crop.size == 0:
            print(f"[{i+1}/{len(image_files)}] 未检测到床: {img_path.name}")
            n_miss += 1
            continue

        x1, y1, x2, y2, conf = det
        out_path = crops_dir / img_path.name
        cv2.imwrite(str(out_path), crop)
        n_found += 1

        if (i + 1) % 100 == 0 or i == 0:
            print(f"[{i+1}/{len(image_files)}] conf={conf:.3f}  "
                  f"bbox=({x1},{y1})-({x2},{y2})  size={x2-x1}x{y2-y1}")

        if not args.no_viz:
            viz = image.copy()
            cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(viz, f"bed {conf:.2f}", (x1, max(y1 - 8, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imwrite(str(viz_dir / f"{img_path.stem}_det.jpg"), viz)

    print(f"\n完成: 检测到 {n_found}，未检测到 {n_miss}")
    print(f"裁剪图保存至: {crops_dir}")


if __name__ == "__main__":
    main()
