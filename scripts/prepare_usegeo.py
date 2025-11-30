import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from uav_3d_benchmark.config import OUTPUT_ROOT, USEGEO_ROOT
from uav_3d_benchmark.datasets.usegeo import UseGeoConfig, export_colmap_files
from uav_3d_benchmark.colmap_pipeline import run_colmap_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strip", default="strip1", help="UseGeo strip name")
    args = parser.parse_args()

    cfg = UseGeoConfig(root=USEGEO_ROOT, strip_name=args.strip)
    work_root = os.path.join(OUTPUT_ROOT, f"usegeo_{args.strip}")
    sparse_known = os.path.join(work_root, "sparse_known")
    export_colmap_files(cfg, sparse_known)
    fused = run_colmap_pipeline(cfg.image_folder, sparse_known, work_root)
    print("UseGeo fused point cloud:", fused)


if __name__ == "__main__":
    main()
