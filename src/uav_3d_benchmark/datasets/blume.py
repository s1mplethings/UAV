import os
import re
import numpy as np

from uav_3d_benchmark.geometry import rotation_matrix_to_quaternion


class BlumeConfig:
    """
    Config for Blume_drone_data_capture_may2021 (RGB).
    You need:
      - images extracted from rgb_images/rgb.zip
      - external poses: merged_blume2_calibrated_external_camera_parameters.txt
      - internal params: merged_blume2_calibrated_internal_camera_parameters.cam
    """

    def __init__(
        self,
        root: str,
        image_folder: str = None,
        pose_file: str = None,
        intr_file: str = None,
    ):
        self.root = root
        self.image_folder = (
            image_folder
            or os.path.join(root, "rgb_images", "rgb")  # assume unzip to rgb_images/rgb/
        )
        params_root = os.path.join(
            root,
            "merged_blume2.zip (Unzipped Files)",
            "merged_blume2",
            "1_initial",
            "params",
        )
        self.pose_file = pose_file or os.path.join(
            params_root, "merged_blume2_calibrated_external_camera_parameters.txt"
        )
        self.intr_file = intr_file or os.path.join(
            params_root, "merged_blume2_calibrated_internal_camera_parameters.cam"
        )


def _parse_intrinsics(intr_path: str):
    focal_mm = None
    sensor_w_mm = None
    sensor_h_mm = None
    width_px = None
    height_px = None
    xpoff_px = None
    ypoff_px = None
    with open(intr_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.upper().startswith("FOCAL"):
                parts = line.split()
                if len(parts) >= 2:
                    focal_mm = float(parts[1])
            if "sensor width" in line:
                nums = re.findall(r"[\d.]+", line)
                if len(nums) >= 2:
                    sensor_w_mm = float(nums[0])
                    sensor_h_mm = float(nums[1])
            if "Image size" in line and "pixel" in line:
                nums = re.findall(r"[\d.]+", line)
                if len(nums) >= 2:
                    width_px = float(nums[0])
                    height_px = float(nums[1])
            if line.upper().startswith("XPOFF"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        xpoff_px = float(parts[1])
                    except ValueError:
                        pass
            if line.upper().startswith("YPOFF"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        ypoff_px = float(parts[1])
                    except ValueError:
                        pass
    if None in (focal_mm, sensor_w_mm, sensor_h_mm, width_px, height_px):
        raise ValueError("Failed to parse intrinsics file.")
    if xpoff_px is None or ypoff_px is None:
        # fallback to 0 offset
        xpoff_px = 0.0
        ypoff_px = 0.0
    px_size_w = sensor_w_mm / width_px
    focal_px = focal_mm / px_size_w
    cx = width_px * 0.5 + xpoff_px
    cy = height_px * 0.5 + ypoff_px
    return {
        "model": "SIMPLE_PINHOLE",
        "width": int(width_px),
        "height": int(height_px),
        "params": [focal_px, cx, cy],
    }


def _euler_opk_to_R(omega_deg, phi_deg, kappa_deg):
    o = np.deg2rad(omega_deg)
    p = np.deg2rad(phi_deg)
    k = np.deg2rad(kappa_deg)
    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(o), -np.sin(o)],
            [0, np.sin(o), np.cos(o)],
        ],
        dtype=float,
    )
    Ry = np.array(
        [
            [np.cos(p), 0, np.sin(p)],
            [0, 1, 0],
            [-np.sin(p), 0, np.cos(p)],
        ],
        dtype=float,
    )
    Rz = np.array(
        [
            [np.cos(k), -np.sin(k), 0],
            [np.sin(k), np.cos(k), 0],
            [0, 0, 1],
        ],
        dtype=float,
    )
    return Rz @ Ry @ Rx


def export_colmap_files(cfg: BlumeConfig, out_sparse_dir: str, camera_id: int = 1):
    os.makedirs(out_sparse_dir, exist_ok=True)
    intr = _parse_intrinsics(cfg.intr_file)
    cameras_txt = os.path.join(out_sparse_dir, "cameras.txt")
    images_txt = os.path.join(out_sparse_dir, "images.txt")
    points3d_txt = os.path.join(out_sparse_dir, "points3D.txt")

    # 写 cameras.txt
    f = intr["params"][0]
    cx = intr["params"][1]
    cy = intr["params"][2]
    with open(cameras_txt, "w") as fcam:
        fcam.write(
            f"{camera_id} SIMPLE_PINHOLE {intr['width']} {intr['height']} {f} {cx} {cy}\n"
        )

    # poses: imageName X Y Z Omega Phi Kappa (deg)
    with open(images_txt, "w") as fimg, open(cfg.pose_file, "r") as pf:
        image_id = 1
        for line in pf:
            line = line.strip()
            if not line or line.startswith("#") or line.lower().startswith("imagename"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            fname = parts[0]
            X = float(parts[1])
            Y = float(parts[2])
            Z = float(parts[3])
            omega = float(parts[4])
            phi = float(parts[5])
            kappa = float(parts[6])
            R_wc = _euler_opk_to_R(omega, phi, kappa)
            C = np.array([X, Y, Z], dtype=float)
            t = -R_wc @ C
            qw, qx, qy, qz = rotation_matrix_to_quaternion(R_wc)
            fimg.write(
                f"{image_id} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {camera_id} {fname}\n\n"
            )
            image_id += 1

    # 空 points3D
    open(points3d_txt, "w").close()
