import os
import numpy as np

from uav_3d_benchmark.geometry import rotation_matrix_to_quaternion


class UseGeoConfig:
    def __init__(self, root: str, strip_name: str):
        self.root = root
        self.strip_name = strip_name

    @property
    def strip_root(self):
        return os.path.join(self.root, self.strip_name)

    @property
    def image_folder(self):
        return os.path.join(self.strip_root, "images")

    @property
    def pose_file(self):
        return os.path.join(self.strip_root, "camera_poses.txt")

    @property
    def intrinsics_file(self):
        return os.path.join(self.strip_root, "camera_intrinsics.txt")


def _parse_numeric_line(path: str):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(",", " ").split()
            return parts
    raise ValueError(f"No numeric line found in {path}")


def load_usegeo_intrinsics(cfg: UseGeoConfig):
    parts = _parse_numeric_line(cfg.intrinsics_file)
    if len(parts) < 5:
        raise ValueError("Expected intrinsics format: c x0 y0 width height")
    c = float(parts[0])
    x0 = float(parts[1])
    y0 = float(parts[2])
    width = int(float(parts[3]))
    height = int(float(parts[4]))
    return {
        "model": "SIMPLE_PINHOLE",
        "width": width,
        "height": height,
        "params": [c, x0, y0],
    }


def euler_omega_phi_kappa_to_R(omega_deg: float, phi_deg: float, kappa_deg: float) -> np.ndarray:
    omega = np.deg2rad(omega_deg)
    phi = np.deg2rad(phi_deg)
    kappa = np.deg2rad(kappa_deg)

    Rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(omega), -np.sin(omega)],
            [0, np.sin(omega), np.cos(omega)],
        ],
        dtype=float,
    )
    Ry = np.array(
        [
            [np.cos(phi), 0, np.sin(phi)],
            [0, 1, 0],
            [-np.sin(phi), 0, np.cos(phi)],
        ],
        dtype=float,
    )
    Rz = np.array(
        [
            [np.cos(kappa), -np.sin(kappa), 0],
            [np.sin(kappa), np.cos(kappa), 0],
            [0, 0, 1],
        ],
        dtype=float,
    )
    return Rz @ Ry @ Rx


def export_colmap_files(cfg: UseGeoConfig, out_sparse_dir: str, camera_id: int = 1):
    os.makedirs(out_sparse_dir, exist_ok=True)
    intr = load_usegeo_intrinsics(cfg)
    cameras_txt = os.path.join(out_sparse_dir, "cameras.txt")
    images_txt = os.path.join(out_sparse_dir, "images.txt")
    points3d_txt = os.path.join(out_sparse_dir, "points3D.txt")

    with open(cameras_txt, "w") as f:
        f.write(
            f"{camera_id} SIMPLE_PINHOLE {intr['width']} {intr['height']} "
            f"{intr['params'][0]} {intr['params'][1]} {intr['params'][2]}\n"
        )

    with open(images_txt, "w") as f:
        with open(cfg.pose_file, "r") as pf:
            for image_id, line in enumerate(pf, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) < 7:
                    continue
                fname = parts[0]
                X0 = float(parts[1])
                Y0 = float(parts[2])
                Z0 = float(parts[3])
                omega = float(parts[4])
                phi = float(parts[5])
                kappa = float(parts[6])
                R_wc = euler_omega_phi_kappa_to_R(omega, phi, kappa)
                C = np.array([X0, Y0, Z0], dtype=float)
                t = -R_wc @ C
                qw, qx, qy, qz = rotation_matrix_to_quaternion(R_wc)
                f.write(f"{image_id} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {camera_id} {fname}\n\n")

    open(points3d_txt, "w").close()
