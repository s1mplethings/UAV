"""
Lightweight SLAM runner hook.

This does not ship a SLAM algorithm; it just provides a thin wrapper to
invoke an external SLAM binary/script with your dataset path and optional
config, and collect its trajectory output.
"""

import os
import subprocess
from typing import List, Optional


def run_cmd(cmd: List[str], cwd: Optional[str] = None):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def run_slam(
    slam_bin: str,
    dataset_path: str,
    output_dir: str,
    extra_args: Optional[List[str]] = None,
    work_dir: Optional[str] = None,
):
    """
    Generic SLAM runner.

    slam_bin: path to your SLAM executable / script (e.g., ORB-SLAM3, DSO, etc.)
    dataset_path: root of the dataset (e.g., EuRoC mav0 folder or image dir)
    output_dir: where to store SLAM results (trajectories, logs)
    extra_args: additional CLI flags for the SLAM binary
    work_dir: optional working directory when launching the process

    Returns: expected trajectory path if produced, otherwise None.
    """
    os.makedirs(output_dir, exist_ok=True)
    cmd = [slam_bin, dataset_path]
    if extra_args:
        cmd.extend(extra_args)
    run_cmd(cmd, cwd=work_dir)

    # Convention: if the SLAM binary writes traj to output_dir/traj.txt, return it.
    traj_path = os.path.join(output_dir, "traj.txt")
    return traj_path if os.path.exists(traj_path) else None
