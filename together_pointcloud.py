import os
import copy
import struct
import collections
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R


def _strip_path(s: str) -> str:
    return s.strip().strip('"').strip("'")


def _find_first_ply(path: str) -> str:
    if os.path.isfile(path) and path.lower().endswith(".ply"):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    cands = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(".ply"):
                cands.append(os.path.join(root, f))
    if not cands:
        raise FileNotFoundError(f"No .ply found under: {path}")
    cands.sort()
    return cands[0]


def load_ply(path: str) -> PlyData:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return PlyData.read(path)


def save_merged_ply(ply_a: PlyData, ply_b: PlyData, out_path: str) -> None:
    va, vb = ply_a["vertex"].data, ply_b["vertex"].data
    common = [n for n in va.dtype.names if n in vb.dtype.names]
    if not common:
        raise ValueError("No common vertex fields between the two PLY files.")
    dtype = [(n, va.dtype[n]) for n in common]
    merged = np.zeros(len(va) + len(vb), dtype=dtype)
    for n in common:
        merged[n][: len(va)] = va[n]
        merged[n][len(va):] = vb[n]
    if os.path.exists(out_path):
        os.remove(out_path)
    PlyData([PlyElement.describe(merged, "vertex")]).write(out_path)


def preprocess_point_cloud(
        ply: PlyData,
        nb_neighbors: int = 20,
        std_ratio: float = 0.5,
        normal_radius: float = 0.1,
        normal_max_nn: int = 30,
):
    v = ply["vertex"].data
    for k in ("x", "y", "z"):
        if k not in v.dtype.names:
            raise ValueError(f"Missing vertex field: {k}")
    pts = np.stack((v["x"], v["y"], v["z"]), axis=-1)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd, indices = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    indices = np.asarray(indices, dtype=np.int64)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn)
    )
    return pcd, indices


def pca_canonical_transform(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    pts = np.asarray(pcd.points)
    if pts.shape[0] < 3:
        raise ValueError("Insufficient points for PCA.")
    c = pts.mean(axis=0)
    cov = np.cov(pts.T)
    evals, evecs = np.linalg.eigh(cov)
    evecs = evecs[:, np.argsort(evals)[::-1]]
    if np.linalg.det(evecs) < 0:
        evecs[:, 2] *= -1
    T = np.eye(4)
    T[:3, :3] = evecs.T
    T[:3, 3] = -evecs.T @ c
    return T


def apply_transform_to_ply(ply: PlyData, T: np.ndarray, indices: np.ndarray) -> PlyData:
    v = ply["vertex"].data[indices].copy()
    pts = np.stack((v["x"], v["y"], v["z"]), axis=-1)
    pts_t = (T[:3, :3] @ pts.T).T + T[:3, 3]
    v["x"], v["y"], v["z"] = pts_t.T
    if {"nx", "ny", "nz"}.issubset(v.dtype.names):
        nrm = np.stack((v["nx"], v["ny"], v["nz"]), axis=-1)
        nrm_t = (T[:3, :3] @ nrm.T).T
        v["nx"], v["ny"], v["nz"] = nrm_t.T
    return PlyData([PlyElement.describe(v, "vertex")], text=ply.text)


def run_pointcloud_merge(front_ply_path: str, back_ply_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    ply_f = load_ply(front_ply_path)
    ply_b = load_ply(back_ply_path)

    pcd_f, idx_f = preprocess_point_cloud(ply_f)
    pcd_b, idx_b = preprocess_point_cloud(ply_b)

    Tf = pca_canonical_transform(pcd_f)
    Tb = pca_canonical_transform(pcd_b)

    pcd_f_pca = copy.deepcopy(pcd_f).transform(Tf)
    pcd_b_pca = copy.deepcopy(pcd_b).transform(Tb)

    pcd_f_down = pcd_f_pca.voxel_down_sample(voxel_size=0.01)
    pcd_b_down = pcd_b_pca.voxel_down_sample(voxel_size=0.01)

    scale = np.percentile(np.linalg.norm(np.asarray(pcd_f_pca.points), axis=1), 90) / np.percentile(
        np.linalg.norm(np.asarray(pcd_b_pca.points), axis=1), 90
    )

    z_f = np.asarray(pcd_f_pca.points)[:, 2]
    z_b = np.asarray(pcd_b_pca.points)[:, 2]
    thick = (
                    float(np.percentile(z_f, 98) - np.percentile(z_f, 2))
                    + float(np.percentile(z_b, 98) - np.percentile(z_b, 2)) * scale
            ) / 2.0
    z_offset = 0.5 * thick

    hypotheses = [
        ("I", np.eye(3)),
        ("Rx(pi)", o3d.geometry.get_rotation_matrix_from_xyz((np.pi, 0, 0))),
        ("Ry(pi)", o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi, 0))),
        ("Rz(pi)", o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.pi))),
    ]

    best = {"fitness": -1.0, "tag": None, "R": None, "T_icp": None}

    for tag, Rm in hypotheses:
        T_init = np.eye(4)
        T_init[:3, :3] = Rm * scale
        T_init[2, 3] = -z_offset
        src = copy.deepcopy(pcd_b_down).transform(T_init)
        try:
            reg = o3d.pipelines.registration.registration_icp(
                src,
                pcd_f_down,
                thick * 2.0,
                np.eye(4),
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            )
        except Exception:
            continue
        if reg.fitness > best["fitness"]:
            best.update({"fitness": reg.fitness, "tag": tag, "R": Rm, "T_icp": reg.transformation})

    if best["tag"] is None:
        raise RuntimeError("Registration failed.")

    T_shift_f = np.eye(4)
    T_shift_f[2, 3] = z_offset
    T_world_front = np.linalg.inv(Tf) @ T_shift_f @ Tf

    T_init_best = np.eye(4)
    T_init_best[:3, :3] = best["R"] * scale
    T_init_best[2, 3] = -z_offset
    T_world_back = np.linalg.inv(Tf) @ best["T_icp"] @ T_init_best @ Tb

    T_back_to_front = np.linalg.inv(T_world_front) @ T_world_back

    out_name = f"Merged_{best['tag']}.ply"
    out_path = os.path.join(out_dir, out_name)

    ply_f_out = apply_transform_to_ply(ply_f, T_world_front, idx_f)
    ply_b_out = apply_transform_to_ply(ply_b, T_world_back, idx_b)
    save_merged_ply(ply_f_out, ply_b_out, out_path)

    return T_back_to_front.astype(np.float64), T_world_front.astype(np.float64), out_path


Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


def qvec2rotmat(qvec):
    return R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()


def rotmat2qvec(mat):
    quat = R.from_matrix(mat).as_quat()
    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)


def read_next_bytes(fid, num_bytes, fmt, endian="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian + fmt, data)


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            cam_id, model_id, width, height = read_next_bytes(fid, 24, "iiQQ")
            model_params_map = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8, 6: 12, 7: 5, 8: 4, 9: 5, 10: 12}
            num_params = model_params_map.get(model_id, 3)
            params = read_next_bytes(fid, 8 * num_params, "d" * num_params)
            cameras[cam_id] = Camera(
                id=cam_id, model=model_id, width=width, height=height, params=np.array(params, dtype=np.float64)
            )
    return cameras


def read_images_binary(path):
    images = {}
    with open(path, "rb") as fid:
        num_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_images):
            props = read_next_bytes(fid, 64, "idddddddi")
            img_id = props[0]
            qvec = np.array(props[1:5], dtype=np.float64)
            tvec = np.array(props[5:8], dtype=np.float64)
            cam_id = props[8]

            name = ""
            ch = read_next_bytes(fid, 1, "c")[0]
            while ch != b"\x00":
                name += ch.decode("utf-8")
                ch = read_next_bytes(fid, 1, "c")[0]

            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            dtype = np.dtype([("x", "f8"), ("y", "f8"), ("pt3d_id", "u8")])
            data = np.frombuffer(fid.read(num_points2D * 24), dtype=dtype)

            images[img_id] = Image(
                id=img_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=cam_id,
                name=name,
                xys=np.stack([data["x"], data["y"]], axis=1),
                point3D_ids=data["pt3d_id"],
            )
    return images


def read_points3D_binary(path):
    points3D = {}
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            props = read_next_bytes(fid, 43, "QdddBBBd")
            pid = props[0]
            xyz = np.array(props[1:4], dtype=np.float64)
            rgb = np.array(props[4:7], dtype=np.uint8)
            err = float(props[7])

            track_len = read_next_bytes(fid, 8, "Q")[0]
            track = read_next_bytes(fid, 8 * track_len, "II" * track_len)

            points3D[pid] = Point3D(
                id=pid,
                xyz=xyz,
                rgb=rgb,
                error=err,
                image_ids=np.array(tuple(map(int, track[0::2])), dtype=np.int64),
                point2D_idxs=np.array(tuple(map(int, track[1::2])), dtype=np.int64),
            )
    return points3D


def write_cameras_binary(cameras, path):
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(cameras)))
        for _, cam in cameras.items():
            fid.write(struct.pack("<iiQQ", cam.id, cam.model, cam.width, cam.height))
            for p in cam.params:
                fid.write(struct.pack("<d", float(p)))


def write_images_binary(images, path):
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(images)))
        for _, img in images.items():
            fid.write(struct.pack("<idddddddi", img.id, *img.qvec, *img.tvec, img.camera_id))
            fid.write(img.name.encode("utf-8") + b"\x00")
            fid.write(struct.pack("<Q", len(img.xys)))
            packed = np.empty(len(img.xys), dtype=[("x", "f8"), ("y", "f8"), ("pt3d_id", "u8")])
            packed["x"] = img.xys[:, 0]
            packed["y"] = img.xys[:, 1]
            packed["pt3d_id"] = img.point3D_ids
            fid.write(packed.tobytes())


def write_points3D_binary(points3D, path):
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(points3D)))
        for _, pt in points3D.items():
            fid.write(struct.pack("<QdddBBBd", pt.id, *pt.xyz, *pt.rgb, float(pt.error)))
            fid.write(struct.pack("<Q", len(pt.image_ids)))
            for i in range(len(pt.image_ids)):
                fid.write(struct.pack("<II", int(pt.image_ids[i]), int(pt.point2D_idxs[i])))


def decompose_transform(T):
    t_vec = T[:3, 3]
    scales = np.linalg.norm(T[:3, :3], axis=0)
    s = np.mean(scales)
    R_raw = T[:3, :3] / s
    U, _, Vt = np.linalg.svd(R_raw)
    R_clean = U @ Vt
    if np.linalg.det(R_clean) < 0:
        U[:, 2] *= -1
        R_clean = U @ Vt
    return s, R_clean, t_vec


def transform_colmap_pose(qvec, tvec, T_matrix):
    R_old = qvec2rotmat(qvec)
    C_old = -R_old.T @ tvec
    C_new = (T_matrix[:3, :3] @ C_old) + T_matrix[:3, 3]
    _, R_trans, _ = decompose_transform(T_matrix)
    R_new = R_old @ R_trans.T
    t_new = -R_new @ C_new
    return rotmat2qvec(R_new), t_new


def run_sparse_merge(front_dir_0: str, back_dir_0: str, out_dir_0: str, T_rel: np.ndarray, T_front_offset: np.ndarray):
    os.makedirs(out_dir_0, exist_ok=True)

    cams_f = read_cameras_binary(os.path.join(front_dir_0, "cameras.bin"))
    imgs_f = read_images_binary(os.path.join(front_dir_0, "images.bin"))
    pts_f = read_points3D_binary(os.path.join(front_dir_0, "points3D.bin"))

    cams_b = read_cameras_binary(os.path.join(back_dir_0, "cameras.bin"))
    imgs_b = read_images_binary(os.path.join(back_dir_0, "images.bin"))
    pts_b = read_points3D_binary(os.path.join(back_dir_0, "points3D.bin"))

    T_front = T_front_offset
    T_back = T_front_offset @ T_rel

    Rf, tf = T_front[:3, :3], T_front[:3, 3]
    Rb, tb = T_back[:3, :3], T_back[:3, 3]

    OFF_CAM = 5_000
    OFF_IMG = 20_000
    OFF_PT = 50_000_000

    merged_cams = {}
    merged_imgs = {}
    merged_pts = {}

    for cid, cam in cams_f.items():
        merged_cams[cid] = cam

    for cid, cam in cams_b.items():
        new_cid = cid + OFF_CAM
        merged_cams[new_cid] = Camera(new_cid, cam.model, cam.width, cam.height, cam.params)

    for iid, img in imgs_f.items():
        qvec_new, tvec_new = transform_colmap_pose(img.qvec, img.tvec, T_front)
        merged_imgs[iid] = Image(
            id=iid,
            qvec=qvec_new,
            tvec=tvec_new,
            camera_id=img.camera_id,
            name=img.name,
            xys=img.xys,
            point3D_ids=img.point3D_ids,
        )

    INVALID_ID = np.uint64(18446744073709551615)
    for iid, img in imgs_b.items():
        new_iid = iid + OFF_IMG
        new_cid = img.camera_id + OFF_CAM
        qvec_new, tvec_new = transform_colmap_pose(img.qvec, img.tvec, T_back)
        p3d_ids = img.point3D_ids.copy()
        valid_mask = p3d_ids != INVALID_ID
        p3d_ids[valid_mask] += OFF_PT
        merged_imgs[new_iid] = Image(
            id=new_iid,
            qvec=qvec_new,
            tvec=tvec_new,
            camera_id=new_cid,
            name=img.name,
            xys=img.xys,
            point3D_ids=p3d_ids,
        )

    for pid, pt in pts_f.items():
        xyz_new = Rf @ pt.xyz + tf
        merged_pts[pid] = Point3D(
            id=pid,
            xyz=xyz_new,
            rgb=pt.rgb,
            error=pt.error,
            image_ids=pt.image_ids,
            point2D_idxs=pt.point2D_idxs,
        )

    for pid, pt in pts_b.items():
        new_pid = pid + OFF_PT
        xyz_new = Rb @ pt.xyz + tb
        new_image_ids = pt.image_ids + OFF_IMG
        merged_pts[new_pid] = Point3D(
            id=new_pid,
            xyz=xyz_new,
            rgb=pt.rgb,
            error=pt.error,
            image_ids=new_image_ids,
            point2D_idxs=pt.point2D_idxs,
        )

    write_cameras_binary(merged_cams, os.path.join(out_dir_0, "cameras.bin"))
    write_images_binary(merged_imgs, os.path.join(out_dir_0, "images.bin"))
    write_points3D_binary(merged_pts, os.path.join(out_dir_0, "points3D.bin"))


def main():
    base = _strip_path(input("Unified_Dataset 路径: "))
    point_front_dir = os.path.join(base, "point_front")
    point_back_dir = os.path.join(base, "point_back")
    point_out_dir = os.path.join(base, "point")

    front_ply = _find_first_ply(point_front_dir)
    back_ply = _find_first_ply(point_back_dir)

    T_rel, T_front_offset, merged_ply_path = run_pointcloud_merge(front_ply, back_ply, point_out_dir)

    front_sparse_0 = os.path.join(base, "sparse_front", "0")
    back_sparse_0 = os.path.join(base, "sparse_back", "0")
    out_sparse_0 = os.path.join(base, "sparse", "0")

    run_sparse_merge(front_sparse_0, back_sparse_0, out_sparse_0, T_rel, T_front_offset)

    print(merged_ply_path)
    print(out_sparse_0)


if __name__ == "__main__":
    main()
