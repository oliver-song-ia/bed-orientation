import json
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse
import glob
import os
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--dir", default="/home/oliver/Documents/bed-orientation/output",
                    help="数据集根目录（含 labels*.json 和 images/）")
parser.add_argument("--n", type=int, default=5, help="随机可视化样本数")
args, _ = parser.parse_known_args()

OUTPUT_DIR = args.dir
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")


def load_labels():
    # 优先读合并后的 labels.json，否则合并所有分片
    merged = os.path.join(OUTPUT_DIR, "labels.json")
    if os.path.exists(merged):
        with open(merged) as f:
            return json.load(f)
    shards = sorted(glob.glob(os.path.join(OUTPUT_DIR, "labels_*_*.json")))
    all_labels = []
    for p in shards:
        with open(p) as f:
            all_labels.extend(json.load(f))
    return all_labels


def draw_top_view(ax, bed_pos, bed_yaw, camera_pos, fov_degrees=90):
    cx, cy = camera_pos[0], camera_pos[1]
    bx, by = bed_pos[0], bed_pos[1]

    # 相机固定朝 +Y
    fwd_x, fwd_y = 0.0, 1.0
    rgt_x, rgt_y = fwd_y, -fwd_x

    rel_x, rel_y = bx - cx, by - cy
    dist_2d = np.sqrt(rel_x**2 + rel_y**2) or 1e-6
    bed_cam_x = rel_x * rgt_x + rel_y * rgt_y
    bed_cam_y = rel_x * fwd_x + rel_y * fwd_y

    margin = max(0.6, dist_2d * 0.35)
    xmin = min(0, bed_cam_x) - margin
    xmax = max(0, bed_cam_x) + margin
    ymin = min(0, bed_cam_y) - margin
    ymax = max(0, bed_cam_y) + margin
    xr, yr = xmax - xmin, ymax - ymin
    if xr > yr:
        pad = (xr - yr) / 2; ymin -= pad; ymax += pad
    else:
        pad = (yr - xr) / 2; xmin -= pad; xmax += pad

    ax.set_aspect('equal')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('X  ← camera left | camera right →', fontsize=9)
    ax.set_ylabel('Y  (camera forward)', fontsize=9)
    ax.set_title('Top View (Camera Frame)', fontsize=12)
    ax.grid(True, alpha=0.3)

    # 相机
    ax.plot(0, 0, 'bo', markersize=12, label='Camera')
    cam_arrow_len = dist_2d * 0.25
    ax.arrow(0, 0, 0, cam_arrow_len,
             head_width=dist_2d * 0.06, head_length=dist_2d * 0.04,
             fc='blue', ec='blue', alpha=0.8)

    # FOV 扇形
    fov_half = np.radians(fov_degrees / 2)
    base_angle = np.pi / 2
    fov_dist = dist_2d * 1.15
    angle1 = base_angle + fov_half
    angle2 = base_angle - fov_half
    ax.plot([0, fov_dist * np.cos(angle1)], [0, fov_dist * np.sin(angle1)],
            'b--', alpha=0.3, linewidth=1)
    ax.plot([0, fov_dist * np.cos(angle2)], [0, fov_dist * np.sin(angle2)],
            'b--', alpha=0.3, linewidth=1)
    thetas = np.linspace(angle2, angle1, 50)
    arc_x = np.concatenate([[0], fov_dist * np.cos(thetas), [0]])
    arc_y = np.concatenate([[0], fov_dist * np.sin(thetas), [0]])
    ax.fill(arc_x, arc_y, color='blue', alpha=0.1)

    # 床
    ax.plot(bed_cam_x, bed_cam_y, 'ro', markersize=10, label='Bed')

    # 朝向箭头：0°=headboard朝相机(+Y)，顺时针为正
    arrow_length = dist_2d * 0.18
    display_angle_rad = np.radians((180 - bed_yaw) % 360)
    ax.arrow(bed_cam_x, bed_cam_y,
             arrow_length * np.cos(display_angle_rad),
             arrow_length * np.sin(display_angle_rad),
             head_width=dist_2d * 0.05, head_length=dist_2d * 0.03,
             fc='red', ec='red', linewidth=2)

    ax.text(bed_cam_x, bed_cam_y + dist_2d * 0.12,
            f'Dist: {dist_2d:.2f}m\nYaw: {bed_yaw:.1f}°',
            ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.legend(loc='upper right', fontsize=9)


def visualize_samples(num_samples=5):
    labels = load_labels()
    selected = random.sample(labels, min(num_samples, len(labels)))

    for idx, sample in enumerate(selected):
        fig = plt.figure(figsize=(16, 5))

        img_path = os.path.join(IMAGES_DIR, sample['image'])
        img = Image.open(img_path)

        bed_pos = sample['bed_location']
        bed_yaw = sample['rotation_yaw_degrees']
        camera_pos = sample['camera_config']['location']
        fov = 2 * np.degrees(np.arctan2(
            sample['camera_config']['width'] / 2,
            sample['camera_config']['fx']
        ))

        ax_img = plt.subplot(1, 2, 1)
        ax_img.imshow(img)
        ax_img.axis('off')
        ax_img.set_title(
            f"Sample {sample['sample_id']}  |  {sample['bed_model']}  |  "
            f"bg={'yes' if sample['has_background'] else 'no'}",
            fontsize=11
        )

        ax_top = plt.subplot(1, 2, 2)
        draw_top_view(ax_top, bed_pos, bed_yaw, camera_pos, fov_degrees=fov)

        plt.tight_layout()
        print(f"Sample {idx+1}/{num_samples}  id={sample['sample_id']}  yaw={bed_yaw:.1f}°")
        plt.show()
        plt.close()


if __name__ == "__main__":
    visualize_samples(args.n)
