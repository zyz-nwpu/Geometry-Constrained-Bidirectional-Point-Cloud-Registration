import os
import argparse
import numpy as np
import cv2
import struct
from plyfile import PlyData, PlyElement
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import re


def parse_args():
    parser = argparse.ArgumentParser(description="根据2D掩码清理Colmap稀疏点云工具")

    parser.add_argument("--input_ply", type=str, required=True,
                        help="输入点云路径 (例如: points3D.ply)")
    parser.add_argument("--colmap_dir", type=str, required=True,
                        help="Colmap稀疏重建结果目录 (包含 cameras.bin, images.bin)")
    parser.add_argument("--mask_dir", type=str, required=True,
                        help="包含掩码图片的文件夹路径")
    parser.add_argument("--output_ply", type=str, required=True,
                        help="输出清理后的点云路径")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="保留阈值 (0.0-1.0). 点必须在多少比例的视角中处于Mask内才会被保留。")
    parser.add_argument("--binary", action='store_true', default=True,
                        help="如果Colmap文件是.bin格式则保留默认，如果是.txt请去掉此勾选(当前代码仅实现.bin读取)")

    return parser.parse_args()


def extract_digits(filename):
    base = os.path.splitext(os.path.basename(filename))[0]
    digits = ''.join(re.findall(r'\d+', base))
    return digits


# =============================================================================
# 简易 Colmap 读取器
# =============================================================================
def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = struct.unpack("<Q", fid.read(8))[0]
        print(f"[DEBUG] cameras.bin 中相机数量: {num_cameras}")

        for _ in range(num_cameras):
            camera_properties = struct.unpack("<iiQQ", fid.read(24))
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]

            if model_id == 0:  # SIMPLE_PINHOLE
                params = struct.unpack("<3d", fid.read(24))
            elif model_id == 1:  # PINHOLE
                params = struct.unpack("<4d", fid.read(32))
            elif model_id == 2:  # SIMPLE_RADIAL
                params = struct.unpack("<4d", fid.read(32))  # f, cx, cy, k
            else:
                print(f"[Warning] 不支持的相机模型ID: {model_id}, 跳过 camera_id={camera_id}")
                continue

            cameras[camera_id] = (width, height, params, model_id)

    return cameras


def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = struct.unpack("<Q", fid.read(8))[0]
        print(f"[DEBUG] images.bin 中图像数量: {num_reg_images}")

        for _ in range(num_reg_images):
            # 正确大小：I(4) + 4d(32) + 3d(24) = 60 bytes
            fmt = "<I4d3d"
            num_bytes = struct.calcsize(fmt)  # 60
            binary_image_properties = struct.unpack(fmt, fid.read(num_bytes))

            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5], dtype=np.float64)
            tvec = np.array(binary_image_properties[5:8], dtype=np.float64)

            camera_id = struct.unpack("<I", fid.read(4))[0]

            name_bytes = bytearray()
            while True:
                char = fid.read(1)
                if char == b"\x00":
                    break
                name_bytes.extend(char)
            name = name_bytes.decode("utf-8")

            num_points2D = struct.unpack("<Q", fid.read(8))[0]
            fid.read(num_points2D * 24)  # 每个点: x(double), y(double), point3D_id(q) = 24 bytes

            images[image_id] = (qvec, tvec, camera_id, name)

    return images


# =============================================================================
# 核心逻辑
# =============================================================================
def get_projection_matrix(qvec, tvec, params, width, height, model_id):
    # Colmap: qvec = [qw, qx, qy, qz]
    # scipy: [x, y, z, w]
    rot = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()

    K = np.eye(3, dtype=np.float64)
    if model_id == 0 or model_id == 2:  # SIMPLE_PINHOLE / SIMPLE_RADIAL
        f, cx, cy = params[0], params[1], params[2]
        K[0, 0] = f
        K[1, 1] = f
        K[0, 2] = cx
        K[1, 2] = cy
    elif model_id == 1:  # PINHOLE
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy

    return rot, tvec, K


def main():
    args = parse_args()

    if not os.path.exists(args.colmap_dir):
        print("错误: Colmap目录不存在")
        return

    if not os.path.exists(args.input_ply):
        print(f"错误: 点云文件不存在: {args.input_ply}")
        return

    if not os.path.exists(args.mask_dir):
        print(f"错误: mask目录不存在: {args.mask_dir}")
        return

    print("正在读取 Colmap 相机参数...")
    cam_bin = os.path.join(args.colmap_dir, "cameras.bin")
    img_bin = os.path.join(args.colmap_dir, "images.bin")

    if os.path.exists(cam_bin) and os.path.exists(img_bin):
        cameras = read_cameras_binary(cam_bin)
        images = read_images_binary(img_bin)
    else:
        print("错误: 未在Colmap目录中找到 .bin 文件。请确保目录包含 cameras.bin 和 images.bin")
        return

    print(f"[DEBUG] 读取到 cameras 数量: {len(cameras)}")
    print(f"[DEBUG] 读取到 images 数量: {len(images)}")
    print(f"[DEBUG] cameras 的前20个 key: {list(cameras.keys())[:20]}")

    print(f"正在读取点云: {args.input_ply}")
    plydata = PlyData.read(args.input_ply)

    vertex = plydata['vertex']
    x = np.asarray(vertex['x'])
    y = np.asarray(vertex['y'])
    z = np.asarray(vertex['z'])
    points_3d = np.stack([x, y, z], axis=1)

    num_points = points_3d.shape[0]
    print(f"点云包含 {num_points} 个点")

    inside_count = np.zeros(num_points, dtype=np.float32)
    seen_count = np.zeros(num_points, dtype=np.float32) + 1e-6

    print("正在扫描 mask 文件...")
    mask_files = os.listdir(args.mask_dir)
    mask_map = {}

    for f in mask_files:
        full_path = os.path.join(args.mask_dir, f)
        if not os.path.isfile(full_path):
            continue

        digit_key = extract_digits(f)
        print(f"[DEBUG] mask file = '{f}', extracted digit_key = '{digit_key}'")

        if digit_key != "":
            if digit_key in mask_map:
                print(f"[Warning] 重复的 mask key: '{digit_key}', 旧文件='{os.path.basename(mask_map[digit_key])}', 新文件='{f}'，将覆盖为新文件")
            mask_map[digit_key] = full_path

    print(f"[DEBUG] 最终 mask_map keys 数量: {len(mask_map)}")
    print(f"[DEBUG] 前20个 mask_map keys: {list(mask_map.keys())[:20]}")

    if len(mask_map) == 0:
        print("错误: 没有从 mask 文件名中提取到任何数字 key，请检查 mask 文件命名")
        return

    print("开始投影计算与掩码校验...")

    matched_image_count = 0
    skipped_no_digit = 0
    skipped_no_mask = 0
    skipped_bad_camera = 0

    for img_id in tqdm(images):
        qvec, tvec, cam_id, name = images[img_id]

        digit_key = extract_digits(name)
        print(f"[DEBUG] image name = '{name}', extracted digit_key = '{digit_key}', camera_id = {cam_id}")

        if digit_key == "":
            print(f"[DEBUG] 跳过: image '{name}' 提取不到数字")
            skipped_no_digit += 1
            continue

        if digit_key not in mask_map:
            print(f"[DEBUG] 未匹配到 mask: image='{name}', digit_key='{digit_key}'")
            print(f"[DEBUG] 当前 mask_map keys 示例: {list(mask_map.keys())[:20]}")
            skipped_no_mask += 1
            continue

        matched_image_count += 1
        mask_path = mask_map[digit_key]
        print(f"[DEBUG] 匹配成功: image='{name}' -> mask='{os.path.basename(mask_path)}'")

        mask = cv2.imread(mask_path, 0)
        if mask is None:
            print(f"[Warning] mask 读取失败: {mask_path}")
            continue

        if cam_id not in cameras:
            print(f"[Warning] image '{name}' 对应的 camera_id={cam_id} 不在 cameras 中，跳过")
            skipped_bad_camera += 1
            continue

        w, h, params, model_id = cameras[cam_id]

        if mask.shape[0] != h or mask.shape[1] != w:
            print(f"[DEBUG] mask 尺寸与相机尺寸不一致: mask={mask.shape[::-1]}, camera=({w}, {h})，将进行 resize")
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        R_mat, t_vec, K = get_projection_matrix(qvec, tvec, params, w, h, model_id)

        points_cam = points_3d @ R_mat.T + t_vec
        valid_z = points_cam[:, 2] > 0.001

        points_2d = points_cam[:, :2] / points_cam[:, 2:]

        u = points_2d[:, 0] * K[0, 0] + K[0, 2]
        v = points_2d[:, 1] * K[1, 1] + K[1, 2]

        valid_uv = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        valid_indices = np.where(valid_z & valid_uv)[0]

        if len(valid_indices) == 0:
            print(f"[DEBUG] image='{name}' 没有任何点投影到有效视野内")
            continue

        sample_u = np.round(u[valid_indices]).astype(int)
        sample_v = np.round(v[valid_indices]).astype(int)

        sample_u = np.clip(sample_u, 0, w - 1)
        sample_v = np.clip(sample_v, 0, h - 1)

        mask_values = mask[sample_v, sample_u]
        is_foreground = mask_values > 127

        seen_count[valid_indices] += 1
        inside_count[valid_indices[is_foreground]] += 1

    print("投影统计结束")
    print(f"[DEBUG] 匹配成功的图片数: {matched_image_count}")
    print(f"[DEBUG] 因提取不到数字而跳过的图片数: {skipped_no_digit}")
    print(f"[DEBUG] 因找不到对应 mask 而跳过的图片数: {skipped_no_mask}")
    print(f"[DEBUG] 因 camera_id 异常而跳过的图片数: {skipped_bad_camera}")

    scores = inside_count / seen_count
    keep_mask = scores > args.threshold

    num_kept = int(np.sum(keep_mask))
    print(f"筛选结果: 保留 {num_kept} / {num_points} ({num_kept / num_points * 100:.2f}%)")

    if num_kept == 0:
        print("警告: 没有点被保留，请检查:")
        print("1. images.bin 是否被正确解析（本版已修正 60 bytes 读取问题）")
        print("2. mask 前景是否为白色 (>127)")
        print("3. 阈值是否过高")
        print("4. 点云和 colmap 模型是否来自同一次重建")
        return

    print(f"正在保存至: {args.output_ply}")

    original_dtype = vertex.data.dtype
    vertex_array = np.empty(num_kept, dtype=original_dtype)

    for field_name in original_dtype.names:
        vertex_array[field_name] = np.asarray(vertex[field_name])[keep_mask]

    el = PlyElement.describe(vertex_array, 'vertex')
    PlyData([el], text=False).write(args.output_ply)

    print("完成！")


if __name__ == "__main__":
    main()