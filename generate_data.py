import math
import random
import json
import os
import sys
import subprocess

# bpy 和 mathutils 只在 Blender 内部 worker 进程中可用
try:
    import bpy
    from mathutils import Matrix, Vector
    IN_BLENDER = True
except ImportError:
    IN_BLENDER = False

# 配置路径
ASSET_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ASSET_DIR, "output")

# 床模型列表 (1-8)
BED_MODELS = [
    os.path.join(ASSET_DIR, "bed_assets", f"{i}.glb")
    for i in range(1, 10)
]

# 背景场景列表
BACKGROUND_SCENES = [
    os.path.join(ASSET_DIR, "scene_assets", f"scene{i}.glb")
    for i in range(1, 5)
]

NUM_SAMPLES = 2000

# 相机配置 (Orbbec Gemini 335L)
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FX = 611.8137817382812
CAMERA_FY = 611.662841796875
CAMERA_CX = 640.3287353515625
CAMERA_CY = 360.26043701171875
HORIZONTAL_FOV = 90  # 度

# 背景比例：20% 样本加载背景场景，80% 无背景
BG_RATIO = 0.2

# 光照方案
LIGHT_SCHEMES = ["hdri", "mixed"]

# 角度约定：
#   0° = 椅子正面朝向相机左侧 (-X 方向)
#   顺时针为正（自上向下看）
#
# 换算依据（假设椅子 GLB 默认正面朝向 -Y，即朝向相机）：
#   Blender Z轴旋转 angle_z（逆时针为正）→ 标签角度 = (angle_z + 90) % 360
#   若椅子模型默认朝向不同，修改此偏移量即可


def reset_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def setup_camera(name, location, pitch_degrees):
    """
    创建并配置相机，朝向 Y 轴正方向。
    pitch_degrees=0 为水平平视，>0 向上仰视，<0 向下俯视。
    """
    bpy.ops.object.camera_add(location=location)
    camera = bpy.context.object
    camera.name = name

    camera.data.sensor_fit = 'HORIZONTAL'
    camera.data.angle = math.radians(HORIZONTAL_FOV)

    # 目标点：正前方 1m，按俯仰角调整 Z
    target_y = location[1] + 1.0
    target_z = location[2] + math.tan(math.radians(pitch_degrees))

    bpy.ops.object.empty_add(location=(location[0], target_y, target_z))
    target = bpy.context.object
    target.name = f"{name}_Target_Temp"

    tt = camera.constraints.new(type='TRACK_TO')
    tt.target = target
    tt.track_axis = 'TRACK_NEGATIVE_Z'
    tt.up_axis = 'UP_Y'

    bpy.context.view_layer.update()
    bpy.context.view_layer.objects.active = camera
    bpy.ops.constraint.apply(constraint=tt.name, owner='OBJECT')
    bpy.data.objects.remove(target, do_unlink=True)

    return camera


def setup_scene():
    """
    初始化场景：创建单个平视相机（pitch=0°）。
    相机高度在 render_single_sample 中每帧随机化（40-60cm）。
    """
    reset_scene()

    # 删除所有现有光源
    for obj in list(bpy.data.objects):
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)
    for light_data in list(bpy.data.lights):
        if light_data.users == 0:
            bpy.data.lights.remove(light_data)

    # 单个平视相机，初始位置占位，后续每帧更新
    camera = setup_camera("Camera_01", (0, 0, 0.5), pitch_degrees=0)
    print("场景初始化完成：单平视相机 (pitch=0°)")
    return camera


def clear_mesh_data():
    """清理所有网格相关数据以释放内存"""
    for obj in list(bpy.data.objects):
        if obj.type == 'MESH':
            bpy.data.objects.remove(obj, do_unlink=True)

    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for material in list(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material)
    for texture in list(bpy.data.textures):
        if texture.users == 0:
            bpy.data.textures.remove(texture)
    for image in list(bpy.data.images):
        if image.users == 0:
            bpy.data.images.remove(image)


def load_bed_only(bed_path):
    """仅加载床，不加载背景场景（80% 的样本）"""
    clear_mesh_data()
    print(f"加载床（无背景）: {os.path.basename(bed_path)}")
    bpy.ops.import_scene.gltf(filepath=bed_path)
    bed = bpy.context.selected_objects[0]
    bed.name = "Bed_Target"
    bed.rotation_mode = 'XYZ'
    return bed


def load_scene_and_bed(scene_path, bed_path):
    """
    加载背景场景 + 床。
    返回 (bed对象, 场景对象名称集合)
    场景对象名称集合用于后续 swap_bed_in_scene 识别哪些对象属于背景。
    """
    clear_mesh_data()

    if os.path.exists(scene_path):
        print(f"加载背景: {os.path.basename(scene_path)}")
        bpy.ops.import_scene.gltf(filepath=scene_path)
    else:
        print(f"背景场景不存在: {scene_path}")

    # 删除场景自带的光源，避免累积
    for obj in list(bpy.data.objects):
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)

    # 记录场景对象名称（导入床前）
    scene_object_names = set(bpy.data.objects.keys())

    print(f"加载床（有背景）: {os.path.basename(bed_path)}")
    bpy.ops.import_scene.gltf(filepath=bed_path)
    bed = bpy.context.selected_objects[0]
    bed.name = "Bed_Target"
    bed.rotation_mode = 'XYZ'
    return bed, scene_object_names


def swap_bed_in_scene(bed_path, scene_object_names):
    """
    在已加载的背景场景中替换床，不重新加载场景。
    删除所有不属于背景场景的非相机/光源对象，再导入新床。
    """
    for obj in list(bpy.data.objects):
        if obj.name not in scene_object_names and obj.type not in ('CAMERA', 'LIGHT'):
            bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    for material in list(bpy.data.materials):
        if material.users == 0:
            bpy.data.materials.remove(material)

    print(f"替换床（有背景）: {os.path.basename(bed_path)}")
    bpy.ops.import_scene.gltf(filepath=bed_path)
    bed = bpy.context.selected_objects[0]
    bed.name = "Bed_Target"
    bed.rotation_mode = 'XYZ'
    return bed


def randomize_lighting(scheme):
    """
    根据方案随机化光照（每帧调用）：
      "point" : 仅点光源，暗色世界背景
      "hdri"  : 仅程序化天空纹理（模拟 HDRI 环境光），无点光源
      "mixed" : 天空纹理 + 点光源
    """
    # 删除所有现有光源
    for obj in list(bpy.data.objects):
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)
    for light_data in list(bpy.data.lights):
        if light_data.users == 0:
            bpy.data.lights.remove(light_data)

    # 设置世界节点
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    if scheme in ("hdri", "mixed"):
        # Preetham 天空纹理模拟室外 HDRI
        sky = nodes.new('ShaderNodeTexSky')
        sky.sky_type = 'PREETHAM'
        sky.turbidity = random.uniform(2.0, 8.0)
        sun_elev = random.uniform(math.radians(10), math.radians(60))
        sun_azimuth = random.uniform(0, 2 * math.pi)
        sky.sun_direction = (
            math.sin(sun_elev) * math.cos(sun_azimuth),
            math.sin(sun_elev) * math.sin(sun_azimuth),
            math.cos(sun_elev)
        )
        bg = nodes.new('ShaderNodeBackground')
        bg.inputs['Strength'].default_value = random.uniform(0.8, 2.0)
        out = nodes.new('ShaderNodeOutputWorld')
        links.new(sky.outputs['Color'], bg.inputs['Color'])
        links.new(bg.outputs['Background'], out.inputs['Surface'])
    else:
        # "point" scheme：深色纯色背景
        bg = nodes.new('ShaderNodeBackground')
        bg.inputs['Color'].default_value = (0.05, 0.05, 0.05, 1.0)
        bg.inputs['Strength'].default_value = 1.0
        out = nodes.new('ShaderNodeOutputWorld')
        links.new(bg.outputs['Background'], out.inputs['Surface'])

    if scheme in ("point", "mixed"):
        # 固定顶光（复用原始脚本逻辑）
        bpy.ops.object.light_add(type='POINT', location=(0, 0, 3.0))
        top_light = bpy.context.object
        top_light.name = "Top_Light_Fixed"
        top_light.data.energy = 300

        # 随机额外点光源 1-3 个（复用原始脚本逻辑）
        num_lights = random.randint(1, 3)
        for i in range(num_lights):
            lx = random.uniform(-2, 2)
            ly = random.uniform(-1, 3)
            lz = random.uniform(1.5, 3.5)
            bpy.ops.object.light_add(type='POINT', location=(lx, ly, lz))
            light = bpy.context.object
            light.name = f"Light_{i+1}"
            light.data.energy = random.uniform(200, 400)

    print(f"  光照方案: {scheme}")


def render_single_sample(camera, bed, sample_idx, scene_name, bed_name, output_dir, has_background):
    """渲染单帧：随机光照、相机高度、床位置和朝向"""

    # 随机光照方案
    scheme = random.choice(LIGHT_SCHEMES)
    randomize_lighting(scheme)

    # 随机相机位置：高度 40-60cm，XY ±1m 偏移
    camera_x = random.uniform(-0.3, 0.3)
    camera_y = random.uniform(-0.3, 0.3)
    camera_z = random.uniform(0.4, 0.6)
    camera.location = (camera_x, camera_y, camera_z)

    bed.scale = (2.0, 2.0, 2.0)

    # 随机床朝向：0-360° 均匀采样
    # 2.glb headboard 方向与其他模型相反，叠加 180° 基础旋转对齐
    angle_z = random.uniform(0, 360)
    base_offset = 180.0 if bed_name == "2.glb" else 0.0
    bed.rotation_euler = (0, 0, math.radians(angle_z + base_offset))

    # 随机床位置（XY），Z 先置 0 后由包围盒计算落地偏移
    bed.location.x = camera_x + random.uniform(-1.0, 1.0)
    bed.location.y = camera_y + random.uniform(1.5, 3.0)
    bed.location.z = 0.0
    bpy.context.view_layer.update()

    # 用世界坐标包围盒确保床底部贴地（等效重力落地）
    bbox_corners = [bed.matrix_world @ Vector(c) for c in bed.bound_box]
    min_z = min(corner.z for corner in bbox_corners)
    bed.location.z -= min_z

    # 各模型额外高度偏移（2, 6, 8 无需调整）
    height_offsets = {"1.glb": 0.7, "2.glb": 0.5, "7.glb": 0.4, "3.glb": 0.5, "4.glb": 0.6, "5.glb": 0.6}
    if bed_name in height_offsets:
        bed.location.z += height_offsets[bed_name]

    bpy.context.view_layer.update()

    # 渲染
    bpy.context.scene.camera = camera
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    file_name = f"bed_{sample_idx:04d}.png"
    file_path = os.path.join(output_dir, "images", file_name)
    bpy.context.scene.render.filepath = file_path
    bpy.ops.render.render(write_still=True)

    # 角度标签换算（与椅子约定相同）：
    #   Blender Z 轴旋转 angle_z（逆时针为正，默认 headboard 朝 -Y）
    #   标签约定：0° = headboard 朝向相机（+Y 方向），顺时针为正
    label_angle = (270 - angle_z) % 360

    label = {
        "sample_id": sample_idx,
        "image": file_name,
        "has_background": has_background,
        "scene": scene_name,
        "bed_model": bed_name,
        "resolution": {
            "width": CAMERA_WIDTH,
            "height": CAMERA_HEIGHT
        },
        "bed_location": list(bed.location),
        "rotation_yaw_degrees": label_angle,
        "rotation_yaw_degrees_raw": angle_z,
        "lighting_scheme": scheme,
        "camera_config": {
            "location": [camera_x, camera_y, camera_z],
            "pitch_degrees": 0,
            "width": CAMERA_WIDTH,
            "height": CAMERA_HEIGHT,
            "fx": CAMERA_FX,
            "fy": CAMERA_FY,
            "cx": CAMERA_CX,
            "cy": CAMERA_CY
        }
    }

    return label


def randomize_and_render(camera, output_dir, total_samples, start_idx=0, end_idx=None):
    """
    数据生成主函数，按背景类型和床/场景分组以减少重复 GLB 加载。

    样本分布：
      索引 0 ~ no_bg_total-1          : 无背景（80%），按床分组
      索引 no_bg_total ~ total-1       : 有背景（20%），按场景分组
    """
    if end_idx is None:
        end_idx = total_samples

    data_labels = []

    bpy.context.scene.render.resolution_x = CAMERA_WIDTH
    bpy.context.scene.render.resolution_y = CAMERA_HEIGHT
    bpy.context.scene.render.resolution_percentage = 100

    n_beds = len(BED_MODELS)
    n_scenes = len(BACKGROUND_SCENES)
    no_bg_total = int(total_samples * (1.0 - BG_RATIO))
    bg_total = total_samples - no_bg_total

    print(f"\n=== 数据生成计划 (样本 {start_idx}-{end_idx-1}) ===")
    print(f"总样本: {total_samples} | 无背景: {no_bg_total} (80%) | 有背景: {bg_total} (20%)")
    print(f"床模型数: {n_beds} | 场景数: {n_scenes}\n")

    # ----------------------------------------------------------------
    # 第一部分：无背景样本（索引 0 ~ no_bg_total-1），按床分组
    # ----------------------------------------------------------------
    samples_per_bed = no_bg_total // n_beds

    for bed_idx, bed_path in enumerate(BED_MODELS):
        group_start = bed_idx * samples_per_bed
        group_end = group_start + samples_per_bed
        if bed_idx == n_beds - 1:
            group_end = no_bg_total  # 最后一组吸收余数

        if group_end <= start_idx or group_start >= end_idx:
            continue

        print(f"[无背景] 组 {bed_idx+1}/{n_beds} 床: {os.path.basename(bed_path)} "
              f"(样本 {group_start}-{group_end-1})")
        bed = load_bed_only(bed_path)
        bed_name = os.path.basename(bed_path)

        for sample_idx in range(group_start, group_end):
            if sample_idx < start_idx or sample_idx >= end_idx:
                continue
            print(f"  渲染样本 {sample_idx+1}/{total_samples}")
            label = render_single_sample(
                camera, bed, sample_idx, None, bed_name, output_dir,
                has_background=False
            )
            data_labels.append(label)

    # ----------------------------------------------------------------
    # 第二部分：有背景样本（索引 no_bg_total ~ total_samples-1）
    # 只按场景分 n_scenes 组，每组内每个样本随机选床并按需替换
    # ----------------------------------------------------------------
    samples_per_scene = bg_total // n_scenes

    for scene_idx, scene_path in enumerate(BACKGROUND_SCENES):
        group_start = no_bg_total + scene_idx * samples_per_scene
        group_end = group_start + samples_per_scene
        if scene_idx == n_scenes - 1:
            group_end = total_samples  # 最后一组吸收余数

        if group_end <= start_idx or group_start >= end_idx:
            continue

        print(f"\n[有背景] 场景: {os.path.basename(scene_path)} "
              f"(样本 {group_start}-{group_end-1}，共 {group_end - group_start} 张)")
        scene_name = os.path.basename(scene_path)

        # 场景只加载一次，先用随机床初始化
        init_bed_path = random.choice(BED_MODELS)
        bed, scene_object_names = load_scene_and_bed(scene_path, init_bed_path)
        current_bed_path = init_bed_path

        for sample_idx in range(group_start, group_end):
            if sample_idx < start_idx or sample_idx >= end_idx:
                continue

            # 每个样本随机选床，换了才替换（只换床，不动场景）
            bed_path = random.choice(BED_MODELS)
            if bed_path != current_bed_path:
                bed = swap_bed_in_scene(bed_path, scene_object_names)
                current_bed_path = bed_path

            bed_name = os.path.basename(bed_path)
            print(f"  渲染样本 {sample_idx+1}/{total_samples} 床: {bed_name}")
            label = render_single_sample(
                camera, bed, sample_idx, scene_name, bed_name, output_dir,
                has_background=True
            )
            data_labels.append(label)

    # 保存分片 JSON（进程标识防冲突）
    json_file = os.path.join(output_dir, f"labels_{start_idx}_{end_idx}.json")
    with open(json_file, 'w') as f:
        json.dump(data_labels, f, indent=4)

    print(f"\n=== 完成 === 生成 {len(data_labels)} 个样本，标签文件: {json_file}")


def merge_json_files(output_dir, total_samples):
    """合并多个进程生成的 JSON 文件"""
    import glob

    json_files = glob.glob(os.path.join(output_dir, "labels_*_*.json"))
    if not json_files:
        print("警告: 没有找到需要合并的 JSON 文件")
        return

    print(f"\n=== 合并 JSON 文件 ({len(json_files)} 个) ===")
    all_labels = []
    for json_file in sorted(json_files):
        with open(json_file, 'r') as f:
            labels = json.load(f)
            all_labels.extend(labels)
        print(f"  已读取: {os.path.basename(json_file)} ({len(labels)} 个样本)")

    all_labels.sort(key=lambda x: x['sample_id'])
    merged_file = os.path.join(output_dir, "labels.json")
    with open(merged_file, 'w') as f:
        json.dump(all_labels, f, indent=4)

    print(f"合并完成: {merged_file} (共 {len(all_labels)} 个样本)")
    for json_file in json_files:
        os.remove(json_file)
    print("已删除分片文件")


def run_parallel(num_workers=4):
    """并行运行多个 Blender 进程"""
    print(f"\n=== 并行数据生成 ({num_workers} 进程, 共 {NUM_SAMPLES} 样本) ===")

    samples_per_worker = NUM_SAMPLES // num_workers
    ranges = []
    for i in range(num_workers):
        start = i * samples_per_worker
        end = (i + 1) * samples_per_worker if i < num_workers - 1 else NUM_SAMPLES
        ranges.append((start, end))
        print(f"Worker {i}: 样本 {start}-{end-1}")

    script_path = os.path.abspath(__file__)
    processes = []
    for i, (start, end) in enumerate(ranges):
        cmd = [
            "blender", "--background",
            "--python", script_path,
            "--", "--start", str(start), "--end", str(end)
        ]
        print(f"启动 Worker {i}: {' '.join(cmd)}")
        proc = subprocess.Popen(cmd)
        processes.append(proc)

    print("\n等待所有 worker 完成...")
    for i, proc in enumerate(processes):
        proc.wait()
        print(f"Worker {i} 已完成 (退出码: {proc.returncode})")

    merge_json_files(OUTPUT_DIR, NUM_SAMPLES)


if __name__ == "__main__":
    import argparse

    if "--" in sys.argv:
        # 单进程模式（由并行启动器调用）
        parser = argparse.ArgumentParser()
        parser.add_argument("--start", type=int, required=True)
        parser.add_argument("--end", type=int, required=True)
        args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])

        os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
        cam = setup_scene()
        randomize_and_render(cam, OUTPUT_DIR, NUM_SAMPLES,
                             start_idx=args.start, end_idx=args.end)
    else:
        # 主进程：并行或单进程模式
        parser = argparse.ArgumentParser(description='床朝向合成数据生成')
        parser.add_argument('--workers', type=int, default=4, help='并行进程数（默认 4）')
        parser.add_argument('--no-parallel', action='store_true', help='禁用并行，单进程运行')
        try:
            args = parser.parse_args()
        except Exception:
            args = argparse.Namespace(workers=4, no_parallel=False)

        os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

        if args.no_parallel:
            print("单进程模式")
            cam = setup_scene()
            randomize_and_render(cam, OUTPUT_DIR, NUM_SAMPLES)
        else:
            run_parallel(num_workers=args.workers)
