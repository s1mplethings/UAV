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
from .video_input import prepare_work_dir_from_video


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SuperPoint + LightGlue (deep-image-matching) + COLMAP MVS 一键 pipeline"
    )
    parser.add_argument(
        "--dir",
        required=False,
        help="数据工作目录（输出目录；默认需要 images/，如提供 --video 会先自动生成 images/）。",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="可选：直接输入视频文件，程序会先抽帧到 <dir>/images 再继续 DIM + COLMAP。",
    )
    parser.add_argument(
        "--video_sample_fps",
        type=float,
        default=2.0,
        help="视频抽帧采样率 FPS（默认 2.0）。",
    )
    parser.add_argument(
        "--video_max_frames",
        type=int,
        default=None,
        help="视频抽帧上限（可空）。",
    )
    parser.add_argument(
        "--video_blur_threshold",
        type=float,
        default=0.0,
        help="可选：视频抽帧时丢弃低于该清晰度分数的候选帧（默认 0，不启用）。",
    )
    parser.add_argument(
        "--video_dedupe_threshold",
        type=float,
        default=0.0,
        help="可选：视频抽帧时丢弃和上一张保留帧过于相似的候选帧（默认 0，不启用）。",
    )
    parser.add_argument(
        "--video_min_gap_sec",
        type=float,
        default=0.0,
        help="可选：视频抽帧时保留帧之间的最小时间间隔秒数（默认 0，不启用）。",
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
        help="测试 pipelines：'all' 或逗号分隔列表（可配合 --test_max_images 限制图片数）。",
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
        help="测试输出目录（默认 <dir>/dim_tests）。",
    )
    parser.add_argument(
        "--test_run_dense",
        action="store_true",
        help="测试时基于每个 pipeline 的匹配结果运行 COLMAP dense，并输出点云。",
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


def _managed_dim_env(args: argparse.Namespace) -> DeepImageMatchingEnv:
    return DeepImageMatchingEnv(env_name=args.dim_env_name)


def _run_wrapper_current_env(argv: list[str]) -> None:
    subprocess.run([sys.executable, "-m", "uav_pipeline.dim_wrapper", *argv], check=True)


def _prepare_video_input(args: argparse.Namespace) -> None:
    prepare_work_dir_from_video(
        work_dir=args.dir,
        video_path=args.video,
        sample_fps=args.video_sample_fps,
        max_frames=args.video_max_frames,
        blur_threshold=args.video_blur_threshold,
        dedupe_threshold=args.video_dedupe_threshold,
        min_gap_sec=args.video_min_gap_sec,
        overwrite=args.overwrite,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.list_dim_pipelines:
        if args.no_dim_env:
            _run_wrapper_current_env(["--list_pipelines"])
        else:
            _managed_dim_env(args).list_pipelines()
        return

    if args.probe_dim_pipelines:
        if not args.dir:
            raise SystemExit("--probe_dim_pipelines 需要 --dir（现有 images/ 工作目录，或用于承接 --video 抽帧的目录）。")
        _prepare_video_input(args)
        if args.no_dim_env:
            _run_wrapper_current_env(
                [
                    "--dir",
                    args.dir,
                    "--pipelines",
                    args.probe_dim_pipelines,
                    "--quality",
                    args.test_quality,
                    "--probe_pipelines",
                    "--print_summary",
                ]
            )
        else:
            _managed_dim_env(args).probe_pipelines(
                scene_dir=args.dir,
                pipelines=args.probe_dim_pipelines,
                quality=args.test_quality,
                gpu=args.gpu,
            )
        return

    if args.test_dim_pipelines:
        if not args.dir:
            raise SystemExit("--test_dim_pipelines 需要 --dir（现有 images/ 工作目录，或用于承接 --video 抽帧的目录）。")
        _prepare_video_input(args)
        if args.no_dim_env:
            cmd = [
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
            if args.test_run_dense:
                cmd.append("--run_dense")
                cmd += ["--colmap_bin", args.colmap_bin]
                if args.patch_match_gpu is not None:
                    cmd += ["--patch_match_gpu", str(args.patch_match_gpu)]
                if args.skip_geom_verification:
                    cmd.append("--skip_geom_verification")
            if args.overwrite:
                cmd.append("--overwrite")
            if args.dim_multi_camera:
                cmd.append("--multi_camera")
            _run_wrapper_current_env(cmd)
        else:
            _managed_dim_env(args).test_pipelines(
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
                run_dense=args.test_run_dense,
                colmap_bin=args.colmap_bin,
                patch_match_gpu=args.patch_match_gpu,
                geom_verification=not args.skip_geom_verification,
                gpu=args.gpu,
            )
        return

    if not args.dir:
        raise SystemExit("需要 --dir（现有 images/ 工作目录，或用于承接 --video 抽帧的目录）。如果只想看可用模型，用 --list_dim_pipelines。")
    if args.video and args.skip_dim:
        raise SystemExit("--video 不能和 --skip_dim 一起使用；视频输入需要先抽帧并运行 DIM。")

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
        video_path=args.video,
        video_sample_fps=args.video_sample_fps,
        video_max_frames=args.video_max_frames,
        video_blur_threshold=args.video_blur_threshold,
        video_dedupe_threshold=args.video_dedupe_threshold,
        video_min_gap_sec=args.video_min_gap_sec,
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
