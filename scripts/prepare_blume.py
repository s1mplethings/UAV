import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
for path in (SRC_ROOT, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from uav_3d_benchmark.datasets.blume import BlumeConfig, export_colmap_files
from uav_3d_benchmark.colmap_pipeline import run_colmap_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="data/Blume_drone_data_capture_may2021",
        help="Blume dataset root",
    )
    parser.add_argument(
        "--image_folder",
        default=None,
        help="Override image folder (default: <root>/rgb_images/rgb)",
    )
    parser.add_argument(
        "--pose_file",
        default=None,
        help="Override pose file (default: merged_blume2_calibrated_external_camera_parameters.txt)",
    )
    parser.add_argument(
        "--intr_file",
        default=None,
        help="Override intrinsics file (default: merged_blume2_calibrated_internal_camera_parameters.cam)",
    )
    parser.add_argument(
        "--output_root",
        default="outputs/blume_merged_blume2",
        help="Output root",
    )
    args = parser.parse_args()

    cfg = BlumeConfig(
        root=args.root,
        image_folder=args.image_folder,
        pose_file=args.pose_file,
        intr_file=args.intr_file,
    )

    work_root = Path(args.output_root)
    sparse_known = work_root / "sparse_known"
    export_colmap_files(cfg, sparse_known)

    fused = run_colmap_pipeline(
        image_folder=cfg.image_folder,
        sparse_known=sparse_known,
        work_root=work_root,
    )
    print("Blume fused point cloud:", fused)


if __name__ == "__main__":
    main()
