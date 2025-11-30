"""
Example stub to run an external SLAM binary on EuRoC.

Usage:
  set SLAM_BIN and SLAM_ARGS to match your SLAM; this script just forwards.
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from uav_3d_benchmark.config import EUROC_ROOT, OUTPUT_ROOT
from uav_3d_benchmark.datasets.euroc import EurocConfig
from uav_3d_benchmark.slam_pipeline import run_slam


def main():
    # TODO: point SLAM_BIN to your executable (e.g., ORB-SLAM3 mono_euroc)
    slam_bin = os.environ.get("SLAM_BIN", r"C:\path\to\your_slam.exe")
    if not os.path.exists(slam_bin):
        raise FileNotFoundError(f"SLAM_BIN not found: {slam_bin}")

    # Example: EuRoC sequence
    cfg = EurocConfig(root=EUROC_ROOT, seq_name="MH_01_easy", cam_id="cam0")
    dataset_path = cfg.seq_root  # pass mav0 folder to your SLAM; adjust as needed

    # Extra args to your SLAM (edit for your tool). For ORB-SLAM3 mono_euroc:
    # slam_bin <vocab> <settings> <sequence_path> <output_traj>
    # Here we just stub a traj output inside work_root.
    work_root = os.path.join(OUTPUT_ROOT, "slam_euroc_MH_01_easy")
    os.makedirs(work_root, exist_ok=True)
    traj_out = os.path.join(work_root, "traj.txt")
    extra_args = os.environ.get("SLAM_ARGS")
    if extra_args:
        args_list = extra_args.split()
    else:
        # Stub: put your default args here
        args_list = [dataset_path, traj_out]

    run_slam(
        slam_bin=slam_bin,
        dataset_path=dataset_path,
        output_dir=work_root,
        extra_args=args_list,
        work_dir=work_root,
    )
    print(f"SLAM finished. Trajectory (if produced): {traj_out}")


if __name__ == "__main__":
    main()
