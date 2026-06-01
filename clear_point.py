import argparse
import re
import struct
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation
from tqdm import tqdm


SUPPORTED_CAMERA_MODELS = {
    0: 3,
    1: 4,
    2: 4,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Purify a dense point cloud using COLMAP camera geometry and foreground masks."
    )
    parser.add_argument("--input_ply", required=True, type=Path)
    parser.add_argument("--colmap_dir", required=True, type=Path)
    parser.add_argument("--mask_dir", required=True, type=Path)
    parser.add_argument("--output_ply", required=True, type=Path)
    parser.add_argument("--threshold", default=0.5, type=float)
    return parser.parse_args()


def numeric_key(path):
    digits = re.findall(r"\d+", Path(path).stem)
    return "".join(digits)


def read_cameras_binary(path):
    cameras = {}
    with path.open("rb") as handle:
        num_cameras = struct.unpack("<Q", handle.read(8))[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = struct.unpack("<iiQQ", handle.read(24))
            if model_id not in SUPPORTED_CAMERA_MODELS:
                raise ValueError(f"Unsupported COLMAP camera model id: {model_id}")
            num_params = SUPPORTED_CAMERA_MODELS[model_id]
            params = struct.unpack("<" + "d" * num_params, handle.read(8 * num_params))
            cameras[camera_id] = {
                "model_id": model_id,
                "width": int(width),
                "height": int(height),
                "params": np.asarray(params, dtype=np.float64),
            }
    return cameras


def read_images_binary(path):
    images = {}
    with path.open("rb") as handle:
        num_images = struct.unpack("<Q", handle.read(8))[0]
        for _ in range(num_images):
            image_id, *pose = struct.unpack("<I4d3d", handle.read(60))
            camera_id = struct.unpack("<I", handle.read(4))[0]

            name_bytes = bytearray()
            while True:
                character = handle.read(1)
                if character == b"\x00":
                    break
                name_bytes.extend(character)

            num_points = struct.unpack("<Q", handle.read(8))[0]
            handle.read(num_points * 24)

            images[image_id] = {
                "qvec": np.asarray(pose[:4], dtype=np.float64),
                "tvec": np.asarray(pose[4:], dtype=np.float64),
                "camera_id": camera_id,
                "name": name_bytes.decode("utf-8"),
            }
    return images


def camera_matrix(params, model_id):
    matrix = np.eye(3, dtype=np.float64)
    if model_id in {0, 2}:
        focal, cx, cy = params[:3]
        matrix[0, 0] = focal
        matrix[1, 1] = focal
        matrix[0, 2] = cx
        matrix[1, 2] = cy
    elif model_id == 1:
        fx, fy, cx, cy = params[:4]
        matrix[0, 0] = fx
        matrix[1, 1] = fy
        matrix[0, 2] = cx
        matrix[1, 2] = cy
    return matrix


def rotation_matrix(qvec):
    return Rotation.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()


def load_masks(mask_dir):
    masks = {}
    for path in sorted(mask_dir.iterdir()):
        if not path.is_file():
            continue
        key = numeric_key(path)
        if key:
            masks[key] = path
    if not masks:
        raise RuntimeError(f"No mask files with numeric stems were found in {mask_dir}")
    return masks


def project_points(points, image, camera):
    rotation = rotation_matrix(image["qvec"])
    translation = image["tvec"]
    intrinsics = camera_matrix(camera["params"], camera["model_id"])

    points_camera = points @ rotation.T + translation
    valid_depth = points_camera[:, 2] > 1e-3
    normalized = points_camera[:, :2] / points_camera[:, 2:3]

    u = normalized[:, 0] * intrinsics[0, 0] + intrinsics[0, 2]
    v = normalized[:, 1] * intrinsics[1, 1] + intrinsics[1, 2]

    width = camera["width"]
    height = camera["height"]
    valid_pixels = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    valid = valid_depth & valid_pixels

    return valid, np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)


def write_filtered_ply(ply_data, keep_mask, output_path):
    vertex = ply_data["vertex"]
    dtype = vertex.data.dtype
    filtered = np.empty(int(keep_mask.sum()), dtype=dtype)
    for name in dtype.names:
        filtered[name] = np.asarray(vertex[name])[keep_mask]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(filtered, "vertex")], text=False).write(output_path)


def validate_inputs(args):
    if not args.input_ply.exists():
        raise FileNotFoundError(args.input_ply)
    if not args.colmap_dir.exists():
        raise FileNotFoundError(args.colmap_dir)
    if not args.mask_dir.exists():
        raise FileNotFoundError(args.mask_dir)
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")


def main():
    args = parse_args()
    validate_inputs(args)

    cameras = read_cameras_binary(args.colmap_dir / "cameras.bin")
    images = read_images_binary(args.colmap_dir / "images.bin")
    masks = load_masks(args.mask_dir)

    ply_data = PlyData.read(args.input_ply)
    vertex = ply_data["vertex"]
    points = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)

    inside_count = np.zeros(points.shape[0], dtype=np.float32)
    visible_count = np.zeros(points.shape[0], dtype=np.float32)
    matched_images = 0

    for image in tqdm(images.values(), desc="mask projection"):
        key = numeric_key(image["name"])
        if not key or key not in masks:
            continue
        camera = cameras.get(image["camera_id"])
        if camera is None:
            continue

        mask = cv2.imread(str(masks[key]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape != (camera["height"], camera["width"]):
            mask = cv2.resize(mask, (camera["width"], camera["height"]), interpolation=cv2.INTER_NEAREST)

        valid, u, v = project_points(points, image, camera)
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size == 0:
            continue

        u_valid = np.clip(u[valid_indices], 0, camera["width"] - 1)
        v_valid = np.clip(v[valid_indices], 0, camera["height"] - 1)
        foreground = mask[v_valid, u_valid] > 127

        visible_count[valid_indices] += 1
        inside_count[valid_indices[foreground]] += 1
        matched_images += 1

    scores = np.divide(
        inside_count,
        np.maximum(visible_count, 1.0),
        out=np.zeros_like(inside_count),
        where=visible_count > 0,
    )
    keep_mask = scores >= args.threshold
    retained = int(keep_mask.sum())

    if retained == 0:
        raise RuntimeError("No points were retained. Check masks, camera model, and threshold.")

    write_filtered_ply(ply_data, keep_mask, args.output_ply)
    print(f"Matched images: {matched_images}")
    print(f"Retained points: {retained} / {points.shape[0]}")
    print(f"Output: {args.output_ply}")


if __name__ == "__main__":
    main()
