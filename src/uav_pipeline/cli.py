"""
Console entrypoint for the deep-image-matching + COLMAP pipeline.

Equivalent to the previous scripts/run_dim_colmap.py logic.
"""

from __future__ import annotations

import argparse
import sys

from .pipeline import PipelineConfig, run_pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SuperPoint + LightGlue (deep-image-matching) + COLMAP MVS 一键 pipeline"
    )
    parser.add_argument(
        "--dir",
        required=True,
        help="数据工作目录（里面要有 images 子目录，deep-image-matching / COLMAP 的所有输出也会放这里）。",
    )
    parser.add_argument(
        "--pipeline",
        default="superpoint+lightglue",
        help="deep-image-matching 的 pipeline 名称，默认 superpoint+lightglue。",
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

    cfg = PipelineConfig(
        work_dir=args.dir,
        pipeline=args.pipeline,
        colmap_bin=args.colmap_bin,
        dense_dir=args.dense_dir,
        gpu=args.gpu,
        patch_match_gpu=args.patch_match_gpu,
        skip_dim=args.skip_dim,
        overwrite=args.overwrite,
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
