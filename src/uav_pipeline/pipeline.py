"""
Reusable pipeline: deep-image-matching (SuperPoint + LightGlue) + COLMAP dense MVS.

This wraps the CLI from ``scripts/run_dim_colmap.py`` into callable functions and
allows passing a custom logger (e.g., GUI text console).
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

LogFn = Callable[[str], None]


def _default_log(msg: str) -> None:
    print(msg)


def run_cmd(cmd: Sequence[str], cwd: Optional[str] = None, log: LogFn = _default_log) -> None:
    """
    Run a shell command, stream stdout/stderr to log, and raise on non-zero exit.
    This makes GUI logs show the real error instead of just returncode.
    """
    log(f"[RUN] {' '.join(str(c) for c in cmd)}")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log(line.rstrip())
    proc.wait()
    if proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def find_sparse_model_dir(work_dir: str, log: LogFn = _default_log) -> str:
    """
    Find a COLMAP sparse model directory that contains cameras.bin and images.bin.

    We search common locations produced by deep-image-matching or COLMAP:
      work_dir / {., colmap, sparse, sparse/0}
    """
    candidates = [
        work_dir,
        os.path.join(work_dir, "colmap"),
        os.path.join(work_dir, "sparse"),
        os.path.join(work_dir, "sparse", "0"),
    ]
    for c in candidates:
        cam = os.path.join(c, "cameras.bin")
        img = os.path.join(c, "images.bin")
        if os.path.exists(cam) and os.path.exists(img):
            log(f"[INFO] Found COLMAP sparse model in: {c}")
            return c
    raise RuntimeError(
        "找不到 COLMAP 稀疏模型（cameras.bin / images.bin）。\n"
        "请确认 deep-image-matching 已经跑完，并且输出在你传入的 --dir 下面。"
    )


@dataclass
class PipelineConfig:
    work_dir: str
    pipeline: str = "superpoint+lightglue"
    colmap_bin: str = "colmap"
    dense_dir: Optional[str] = None
    gpu: Optional[int] = None
    patch_match_gpu: Optional[int] = None
    skip_dim: bool = False
    overwrite: bool = False


def run_dim(cfg: PipelineConfig, log: LogFn = _default_log) -> None:
    """Run deep-image-matching with the requested pipeline and GPU settings."""
    cmd = [
        sys.executable,
        "-m",
        "deep_image_matching",
        "--dir",
        cfg.work_dir,
        "--pipeline",
        cfg.pipeline,
    ]

    if cfg.overwrite:
        cmd += ["--overwrite"]

    if cfg.gpu is not None:
        cmd += ["--device", f"cuda:{cfg.gpu}"]

    run_cmd(cmd, log=log)


def run_colmap_mvs(cfg: PipelineConfig, log: LogFn = _default_log) -> str:
    """Run COLMAP dense reconstruction (undistort → patch_match_stereo → stereo_fusion)."""
    colmap_bin = cfg.colmap_bin
    work_dir = cfg.work_dir

    images_dir = os.path.join(work_dir, "images")
    if not os.path.isdir(images_dir):
        raise RuntimeError(f"找不到 images 目录: {images_dir}")

    sparse_dir = find_sparse_model_dir(work_dir, log=log)

    dense_root = cfg.dense_dir or os.path.join(work_dir, "dense")
    os.makedirs(dense_root, exist_ok=True)

    # 1. image_undistorter
    undistort_dir = dense_root  # COLMAP 默认：输出目录本身作为 dense workspace
    cmd_undistort = [
        colmap_bin,
        "image_undistorter",
        "--image_path",
        images_dir,
        "--input_path",
        sparse_dir,
        "--output_path",
        undistort_dir,
        "--output_type",
        "COLMAP",
    ]
    run_cmd(cmd_undistort, log=log)

    # 2. patch_match_stereo
    cmd_patch_match = [
        colmap_bin,
        "patch_match_stereo",
        "--workspace_path",
        undistort_dir,
        "--workspace_format",
        "COLMAP",
        "--PatchMatchStereo.geom_consistency",
        "true",
    ]
    if cfg.patch_match_gpu is not None:
        cmd_patch_match += [
            "--PatchMatchStereo.gpu_index",
            str(cfg.patch_match_gpu),
        ]
    run_cmd(cmd_patch_match, log=log)

    # 3. stereo_fusion
    fused_path = os.path.join(dense_root, "fused.ply")
    cmd_fusion = [
        colmap_bin,
        "stereo_fusion",
        "--workspace_path",
        undistort_dir,
        "--workspace_format",
        "COLMAP",
        "--input_type",
        "geometric",
        "--output_path",
        fused_path,
    ]
    run_cmd(cmd_fusion, log=log)

    log(f"[OK] Dense point cloud: {fused_path}")
    return fused_path


def run_pipeline(cfg: PipelineConfig, log: LogFn = _default_log) -> str:
    """
    Run the full pipeline (DIM + COLMAP MVS) and return fused.ply path.

    Set cfg.skip_dim to reuse an existing sparse model without rerunning DIM.
    """
    work_dir = os.path.abspath(cfg.work_dir)
    if not os.path.isdir(work_dir):
        raise RuntimeError(f"工作目录不存在: {work_dir}")

    cfg.work_dir = work_dir

    log(f"[INFO] 工作目录: {work_dir}")
    log(f"[INFO] deep-image-matching pipeline: {cfg.pipeline}")
    log(f"[INFO] COLMAP bin: {cfg.colmap_bin}")

    if not cfg.skip_dim:
        log("\n===== Step 1: deep-image-matching (SuperPoint + LightGlue) =====")
        run_dim(cfg, log=log)
    else:
        log("\n[INFO] 跳过 deep-image-matching，直接使用已有的 COLMAP 稀疏结果。")

    log("\n===== Step 2: COLMAP Dense MVS (image_undistorter + patch_match_stereo + stereo_fusion) =====")
    return run_colmap_mvs(cfg, log=log)
