import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
for path in (SRC_ROOT, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from uav_3d_benchmark.config import EUROC_ROOT, OUTPUT_ROOT
from uav_3d_benchmark.datasets.euroc import EurocConfig, export_colmap_files
from uav_3d_benchmark.colmap_pipeline import run_colmap_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="MH_01_easy", help="EuRoC sequence name")
    parser.add_argument("--cam", default="cam0", help="Camera id (cam0 or cam1)")
    args = parser.parse_args()

    cfg = EurocConfig(root=EUROC_ROOT, seq_name=args.seq, cam_id=args.cam)
    work_root = os.path.join(OUTPUT_ROOT, f"euroc_{args.seq}")
    sparse_known = os.path.join(work_root, "sparse_known")
    export_colmap_files(cfg, sparse_known)
    fused = run_colmap_pipeline(cfg.image_folder, sparse_known, work_root)
    print("EuRoC fused point cloud:", fused)


if __name__ == "__main__":
    main()
