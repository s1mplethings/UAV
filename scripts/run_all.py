import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
for path in (SRC_ROOT, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from uav_3d_benchmark.config import EUROC_ROOT, USEGEO_ROOT, OUTPUT_ROOT
from uav_3d_benchmark.datasets.euroc import EurocConfig, export_colmap_files as euroc_export
from uav_3d_benchmark.datasets.usegeo import UseGeoConfig, export_colmap_files as usegeo_export
from uav_3d_benchmark.colmap_pipeline import run_colmap_pipeline


def run_euroc(seq_name="MH_01_easy", cam_id="cam0"):
    cfg = EurocConfig(root=EUROC_ROOT, seq_name=seq_name, cam_id=cam_id)
    work_root = os.path.join(OUTPUT_ROOT, f"euroc_{seq_name}")
    sparse_known = os.path.join(work_root, "sparse_known")
    euroc_export(cfg, sparse_known)
    fused = run_colmap_pipeline(cfg.image_folder, sparse_known, work_root)
    print("EuRoC fused point cloud:", fused)


def run_usegeo(strip_name="strip1"):
    cfg = UseGeoConfig(root=USEGEO_ROOT, strip_name=strip_name)
    work_root = os.path.join(OUTPUT_ROOT, f"usegeo_{strip_name}")
    sparse_known = os.path.join(work_root, "sparse_known")
    usegeo_export(cfg, sparse_known)
    fused = run_colmap_pipeline(cfg.image_folder, sparse_known, work_root)
    print("UseGeo fused point cloud:", fused)


if __name__ == "__main__":
    run_euroc()
    run_usegeo()
