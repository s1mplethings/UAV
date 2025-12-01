#!/usr/bin/env python
r"""
Minimal runner for ETH3D undistorted datasets with known poses.

Usage:
  python scripts/run_eth3d.py            # 使用默认根目录和场景
  python scripts/run_eth3d.py <scene>    # 指定场景名

默认假设结构（以 delivery_area 为例，可根据需要调整 DEFAULT_*）:
  D:\.py_projects\UAV\UAV\data\delivery_area\
    images\dslr_images_undistorted\*.jpg
    dslr_calibration_undistorted\cameras.txt
    dslr_calibration_undistorted\images.txt
    dslr_calibration_undistorted\points3D.txt
输出：D:\.py_projects\UAV\UAV\outputs\<scene>\dense\fused.ply
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

# 默认路径，可按需修改
DEFAULT_DATASETS_ROOT = Path(r"D:\.py_projects\UAV\UAV\data\delivery_area")
DEFAULT_OUTPUT_ROOT = Path(r"D:\.py_projects\UAV\UAV\outputs")
DEFAULT_COLMAP_BIN = r"C:\Program Files\COLMAP\bin\colmap.exe"


def run_cmd(cmd):
    print(" ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def run_eth3d_scene(scene_name: str, datasets_root: Path, output_root: Path, colmap_bin: str):
    # 场景根：若 scene_name 为空，直接用 datasets_root，否则拼子目录
    scene_root = datasets_root if scene_name == "" else datasets_root / scene_name

    # ETH3D delivery_area: images.txt 使用相对路径 "dslr_images_undistorted/XXX"
    # 因此 image_path 直接指向 scene_root，让 COLMAP 自行拼接子目录
    image_path = scene_root
    if not image_path.is_dir():
        raise FileNotFoundError(f"找不到场景目录: {image_path}")

    # 模型目录：优先 dslr_calibration_undistorted，否则用 scene_root
    model_root = scene_root / "dslr_calibration_undistorted"
    if not model_root.is_dir():
        model_root = scene_root

    work_root = output_root / (scene_name if scene_name else scene_root.name)
    dense_root = work_root / "dense"
    if dense_root.exists():
        shutil.rmtree(dense_root)
    dense_root.mkdir(parents=True, exist_ok=True)

    # 1. 用已有 sparse model 做去畸变并生成稠密工作空间
    run_cmd(
        [
            colmap_bin,
            "image_undistorter",
            "--image_path",
            str(image_path),
            "--input_path",
            str(model_root),
            "--output_path",
            str(dense_root),
        ]
    )

    # 2. PatchMatch Stereo
    run_cmd(
        [
            colmap_bin,
            "patch_match_stereo",
            "--workspace_path",
            str(dense_root),
            "--PatchMatchStereo.filter",
            "1",
        ]
    )

    # 3. Stereo Fusion
    fused_ply = dense_root / "fused.ply"
    run_cmd(
        [
            colmap_bin,
            "stereo_fusion",
            "--workspace_path",
            str(dense_root),
            "--output_path",
            str(fused_ply),
        ]
    )
    print("完成, 输出稠密点云:", fused_ply)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scene", nargs="?", default="", help="场景目录名，留空时直接使用 datasets_root")
    parser.add_argument("--datasets_root", default=str(DEFAULT_DATASETS_ROOT), help="数据根目录")
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT), help="输出根目录")
    parser.add_argument("--colmap_bin", default=DEFAULT_COLMAP_BIN, help="colmap 可执行文件路径")
    args = parser.parse_args()

    run_eth3d_scene(
        scene_name=args.scene,
        datasets_root=Path(args.datasets_root),
        output_root=Path(args.output_root),
        colmap_bin=args.colmap_bin,
    )


if __name__ == "__main__":
    main()
