import csv
import os
import bisect
from typing import Dict, List, Tuple

import numpy as np
import yaml

from uav_3d_benchmark.geometry import quaternion_to_rotation_matrix, rotation_matrix_to_quaternion


class EurocConfig:
    def __init__(self, root: str, seq_name: str, cam_id: str = "cam0"):
        self.root = root
        self.seq_name = seq_name
        self.cam_id = cam_id

    @property
    def seq_root(self):
        return os.path.join(self.root, self.seq_name, "mav0")

    @property
    def cam_folder(self):
        return os.path.join(self.seq_root, self.cam_id)

    @property
    def cam_data_csv(self):
        return os.path.join(self.cam_folder, "data.csv")

    @property
    def cam_sensor_yaml(self):
        return os.path.join(self.cam_folder, "sensor.yaml")

    @property
    def gt_csv(self):
        return os.path.join(self.seq_root, "state_groundtruth_estimate0", "data.csv")

    @property
    def image_folder(self):
        return os.path.join(self.cam_folder, "data")


def load_sensor_yaml(cfg: EurocConfig) -> Dict:
    with open(cfg.cam_sensor_yaml, "r") as f:
        return yaml.safe_load(f)


def load_euroc_intrinsics(sensor_yaml: Dict):
    model = sensor_yaml.get("camera_model", "PINHOLE")
    if "image_width" in sensor_yaml and "image_height" in sensor_yaml:
        width = sensor_yaml["image_width"]
        height = sensor_yaml["image_height"]
    elif "resolution" in sensor_yaml:
        # EuRoC uses [width, height]
        width, height = sensor_yaml["resolution"]
    else:
        raise KeyError("sensor.yaml missing image size (image_width/height or resolution)")
    fx, fy, cx, cy = sensor_yaml["intrinsics"]
    return {
        "model": "PINHOLE" if model.lower() == "pinhole" else "SIMPLE_PINHOLE",
        "width": width,
        "height": height,
        "params": [fx, fy, cx, cy] if model.lower() == "pinhole" else [fx, cx, cy],
    }


def load_body_to_cam(sensor_yaml: Dict) -> np.ndarray:
    T = sensor_yaml.get("T_BS") or sensor_yaml.get("T_B_C") or sensor_yaml.get("T_bc")
    if T is None:
        return np.eye(4)
    if isinstance(T, dict) and "data" in T:
        mat = np.array(T["data"], dtype=float).reshape(4, 4)
        return mat
    mat = np.array(T, dtype=float)
    if mat.shape == (4, 4):
        return mat
    raise ValueError("Unsupported T_BS format in sensor.yaml")


def load_euroc_image_list(cfg: EurocConfig) -> List[Tuple[int, str]]:
    imgs = []
    with open(cfg.cam_data_csv, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            ts = int(row[0])
            fname = row[1]
            imgs.append((ts, fname))
    return imgs


def load_euroc_groundtruth(cfg: EurocConfig) -> Dict[int, Dict[str, Tuple[float, float, float]]]:
    gt = {}
    with open(cfg.gt_csv, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            ts = int(row[0])
            px = float(row[1])
            py = float(row[2])
            pz = float(row[3])
            qw = float(row[4])
            qx = float(row[5])
            qy = float(row[6])
            qz = float(row[7])
            gt[ts] = dict(p=(px, py, pz), q=(qw, qx, qy, qz))
    return gt


def compose_world_to_cam(gt_pose: Dict, T_body_cam: np.ndarray = None):
    T_body_cam = np.eye(4) if T_body_cam is None else np.asarray(T_body_cam, dtype=float)
    R_bc = T_body_cam[:3, :3]
    t_bc = T_body_cam[:3, 3]

    qw, qx, qy, qz = gt_pose["q"]
    px, py, pz = gt_pose["p"]
    R_wb = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    p_wb = np.array([px, py, pz], dtype=float)
    t_wb = -R_wb @ p_wb

    R_wc = R_wb @ R_bc
    t_wc = R_wb @ t_bc + t_wb
    qw_c, qx_c, qy_c, qz_c = rotation_matrix_to_quaternion(R_wc)
    return qw_c, qx_c, qy_c, qz_c, float(t_wc[0]), float(t_wc[1]), float(t_wc[2])


def _nearest_gt(ts: int, gt_times: List[int], gt: Dict[int, Dict], max_diff_ns: int = 20_000_000):
    idx = bisect.bisect_left(gt_times, ts)
    candidates = []
    if idx < len(gt_times):
        candidates.append(gt_times[idx])
    if idx > 0:
        candidates.append(gt_times[idx - 1])
    if not candidates:
        return None
    best = min(candidates, key=lambda t: abs(t - ts))
    if abs(best - ts) <= max_diff_ns:
        return gt[best]
    return None


def export_colmap_files(cfg: EurocConfig, out_sparse_dir: str, camera_id: int = 1):
    os.makedirs(out_sparse_dir, exist_ok=True)
    sensor_yaml = load_sensor_yaml(cfg)
    intr = load_euroc_intrinsics(sensor_yaml)
    T_body_cam = load_body_to_cam(sensor_yaml)
    imgs = load_euroc_image_list(cfg)
    gt = load_euroc_groundtruth(cfg)
    gt_times = sorted(gt.keys())

    cameras_txt = os.path.join(out_sparse_dir, "cameras.txt")
    images_txt = os.path.join(out_sparse_dir, "images.txt")
    points3d_txt = os.path.join(out_sparse_dir, "points3D.txt")

    with open(cameras_txt, "w") as f:
        if intr["model"] == "PINHOLE":
            fx, fy, cx, cy = intr["params"]
            f.write(f"{camera_id} PINHOLE {intr['width']} {intr['height']} {fx} {fy} {cx} {cy}\n")
        else:
            f.write(
                f"{camera_id} SIMPLE_PINHOLE {intr['width']} {intr['height']} "
                f"{intr['params'][0]} {intr['params'][1]} {intr['params'][2]}\n"
            )

    with open(images_txt, "w") as f:
        image_id = 1
        for ts, fname in imgs:
            pose = gt.get(ts)
            if pose is None:
                pose = _nearest_gt(ts, gt_times, gt)
            if pose is None:
                continue
            qw, qx, qy, qz, tx, ty, tz = compose_world_to_cam(pose, T_body_cam)
            f.write(f"{image_id} {qw} {qx} {qy} {qz} {tx} {ty} {tz} {camera_id} {fname}\n\n")
            image_id += 1

    open(points3d_txt, "w").close()
