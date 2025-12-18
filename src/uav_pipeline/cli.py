"""
Console entrypoint for the deep-image-matching + COLMAP pipeline.

Equivalent to the previous scripts/run_dim_colmap.py logic.
"""

from __future__ import annotations

import argparse
import subprocess
import sys

from .dim_env import DeepImageMatchingEnv
from .pipeline import PipelineConfig, run_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SuperPoint + LightGlue (deep-image-matching) + COLMAP MVS 一键 pipeline"
    )
    parser.add_argument(
        "--dir",
        required=False,
        help="数据工作目录（里面要有 images 子目录，deep-image-matching / COLMAP 的所有输出也会放这里）。",
    )
    parser.add_argument(
        "--pipeline",
        default="superpoint+lightglue",
        help="deep-image-matching 的 pipeline 名称，默认 superpoint+lightglue。",
    )
    parser.add_argument(
        "--dim_quality",
        default="medium",
        choices=["highest", "high", "medium", "low", "lowest"],
        help="deep-image-matching 的分辨率预设（默认 medium，可选 highest/high/medium/low/lowest）。",
    )
    parser.add_argument(
        "--dim_camera_model",
        default="simple-radial",
        choices=["simple-pinhole", "pinhole", "simple-radial", "opencv"],
        help="写入 COLMAP 数据库时使用的相机模型（默认 simple-radial）。",
    )
    parser.add_argument(
        "--dim_multi_camera",
        action="store_true",
        help="将每张图当作独立相机（一般单相机 UAV 不建议打开；默认是 single_camera）。",
    )
    parser.add_argument(
        "--skip_geom_verification",
        action="store_true",
        help="跳过 COLMAP geometric_verification（不建议，可能导致 mapper 质量变差）。",
    )
    parser.add_argument(
        "--list_dim_pipelines",
        action="store_true",
        help="列出 deep-image-matching 内置的 pipelines，然后退出（用于选择 --pipeline）。",
    )
    parser.add_argument(
        "--probe_dim_pipelines",
        default=None,
        help="探测 pipelines 是否能初始化：'all' 或逗号分隔列表（不跑匹配，仅检查依赖/权重加载）。",
    )
    parser.add_argument(
        "--test_dim_pipelines",
        default=None,
        help="跑一个小规模 smoke test：'all' 或逗号分隔列表（只用前 N 张图）。",
    )
    parser.add_argument(
        "--test_max_images",
        type=int,
        default=None,
        help="测试时只用 images/ 的前 N 张图（留空则使用全部）。",
    )
    parser.add_argument(
        "--test_quality",
        default="low",
        choices=["highest", "high", "medium", "low", "lowest"],
        help="测试用的 DIM 分辨率预设（默认 low）。",
    )
    parser.add_argument(
        "--test_output_dir",
        default=None,
        help="smoke test 输出目录（默认 <dir>/dim_tests）。",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="在 test_dim_pipelines 时记录耗时/内存/GPU，并输出 benchmark.csv/json。",
    )
    parser.add_argument(
        "--benchmark_interval",
        type=float,
        default=0.2,
        help="benchmark RSS 采样间隔秒数（默认 0.2）。",
    )
    parser.add_argument(
        "--no_dim_env",
        action="store_true",
        help="Disable the managed Python 3.9 deep-image-matching environment; run DIM in the current Python env.",
    )
    parser.add_argument(
        "--dim_env_name",
        default="py39_dim_env",
        help="Folder name for the managed deep-image-matching env (default: py39_dim_env).",
    )
    parser.add_argument(
        "--colmap_bin",
        default="colmap",
        help="COLMAP 可执行文件路径，例如 'C:/Program Files/COLMAP/bin/colmap.exe'。",
    )
    parser.add_argument(
        "--dense_dir",
        default=None,
        help="COLMAP dense 重建输出目录，默认为 <dir>/dense。",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="deep-image-matching 用的 GPU index，例如 0。留空则用默认。",
    )
    parser.add_argument(
        "--patch_match_gpu",
        type=int,
        default=None,
        help="COLMAP patch_match_stereo 用的 GPU index，例如 0。留空则由 COLMAP 自己决定。",
    )
    parser.add_argument(
        "--skip_dim",
        action="store_true",
        help="跳过 deep-image-matching，只跑后面的 COLMAP MVS（你已经提前跑完 DIM 的时候用）。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="传给 deep-image-matching 的 --overwrite，强制覆盖已有输出。",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_dim_pipelines:
        if args.no_dim_env:
            subprocess.run([sys.executable, "-m", "uav_pipeline.dim_wrapper", "--list_pipelines"], check=True)
        else:
            DeepImageMatchingEnv().list_pipelines()
        return

    if args.probe_dim_pipelines:
        if not args.dir:
            raise SystemExit("--probe_dim_pipelines 需要 --dir（包含 images/ 的工作目录）。")
        if args.no_dim_env:
            cmd = [
                sys.executable,
                "-m",
                "uav_pipeline.dim_wrapper",
                "--dir",
                args.dir,
                "--pipelines",
                args.probe_dim_pipelines,
                "--quality",
                args.test_quality,
                "--probe_pipelines",
                "--print_summary",
            ]
            subprocess.run(cmd, check=True)
        else:
            DeepImageMatchingEnv().probe_pipelines(
                scene_dir=args.dir,
                pipelines=args.probe_dim_pipelines,
                quality=args.test_quality,
                gpu=args.gpu,
            )
        return

    if args.test_dim_pipelines:
        if not args.dir:
            raise SystemExit("--test_dim_pipelines 需要 --dir（包含 images/ 的工作目录）。")
        if args.no_dim_env:
            cmd = [
                sys.executable,
                "-m",
                "uav_pipeline.dim_wrapper",
                "--dir",
                args.dir,
                "--pipelines",
                args.test_dim_pipelines,
                "--quality",
                args.test_quality,
                "--camera_model",
                args.dim_camera_model,
                "--print_summary",
            ]
            if args.benchmark:
                cmd.append("--benchmark")
                cmd += ["--benchmark_interval", str(args.benchmark_interval)]
            if args.test_max_images is not None:
                cmd += ["--max_images", str(args.test_max_images)]
            if args.test_output_dir:
                cmd += ["--output", args.test_output_dir]
            if args.overwrite:
                cmd.append("--overwrite")
            if args.dim_multi_camera:
                cmd.append("--multi_camera")
            subprocess.run(cmd, check=True)
        else:
            DeepImageMatchingEnv().test_pipelines(
                scene_dir=args.dir,
                pipelines=args.test_dim_pipelines,
                output_dir=args.test_output_dir,
                max_images=args.test_max_images,
                quality=args.test_quality,
                benchmark=args.benchmark,
                benchmark_interval=args.benchmark_interval,
                overwrite=args.overwrite,
                single_camera=not args.dim_multi_camera,
                camera_model=args.dim_camera_model,
                gpu=args.gpu,
            )
        return

    if not args.dir:
        raise SystemExit("需要 --dir（包含 images/ 的工作目录）。如果只想看可用模型，用 --list_dim_pipelines。")

    cfg = PipelineConfig(
        work_dir=args.dir,
        pipeline=args.pipeline,
        colmap_bin=args.colmap_bin,
        dense_dir=args.dense_dir,
        gpu=args.gpu,
        patch_match_gpu=args.patch_match_gpu,
        skip_dim=args.skip_dim,
        overwrite=args.overwrite,
        use_dim_env=not args.no_dim_env,
        dim_env_name=args.dim_env_name,
        dim_quality=args.dim_quality,
        dim_single_camera=not args.dim_multi_camera,
        dim_camera_model=args.dim_camera_model,
        geom_verification=not args.skip_geom_verification,
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
