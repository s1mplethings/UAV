"""
Minimal wrapper to run deep-image-matching as a CLI-friendly script.

This uses the deep_image_matching library directly (since the package ships
without a __main__) and exports a COLMAP database from the produced features
and matches. It does *not* run COLMAP mapper; the caller can do that after this
script finishes (our pipeline does it in Python).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

PIPELINE_ALIASES: dict[str, dict] = {
    # Commonly used combo but not shipped as a built-in DIM pipeline name.
    "sift+lightglue": {
        "base_pipeline": "sift+kornia_matcher",
        "config": {"matcher": {"name": "lightglue"}},
    },
}

SUPERGLUE_OUTDOOR_URL = (
    "https://github.com/magicleap/SuperGluePretrainedNetwork/raw/master/models/weights/superglue_outdoor.pth"
)


def export_matches_to_colmap_db(
    *,
    img_dir: Path,
    feature_path: Path,
    match_path: Path,
    database_path: Path,
    single_camera: bool,
    camera_model: str,
) -> None:
    """
    Export DIM keypoints + matches to a COLMAP sqlite database.

    Important: we insert matches into the `matches` table (not `two_view_geometries`),
    so COLMAP can run `geometric_verification` to estimate the relative geometry.
    """
    if database_path.exists():
        database_path.unlink()

    from deep_image_matching.io.h5_to_db import add_keypoints, add_raw_matches
    from deep_image_matching.utils.database import COLMAPDatabase

    camera_options = {"general": {"single_camera": single_camera, "camera_model": camera_model}}

    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    fname_to_id = add_keypoints(db, feature_path, img_dir, camera_options)
    add_raw_matches(db, match_path, fname_to_id)
    db.commit()


def list_pipelines() -> list[str]:
    try:
        from deep_image_matching.config import confs as dim_confs
        builtins = set(dim_confs.keys())
    except Exception:
        builtins = set()
    return sorted(builtins | set(PIPELINE_ALIASES.keys()))


def _resolve_pipeline(pipeline: str, temp_root: Path) -> tuple[str, Path | None]:
    """
    Resolve a user-facing pipeline name to a DIM pipeline + optional YAML override config.
    Returns (base_pipeline, config_file_path_or_None).
    """
    # Avoid importing deep_image_matching at module import time.
    # For built-in pipelines we don't validate here; DIM's Config will validate later.
    if pipeline not in PIPELINE_ALIASES:
        return pipeline, None

    alias = PIPELINE_ALIASES[pipeline]
    base = alias["base_pipeline"]
    cfg = alias.get("config", {})
    temp_root.mkdir(parents=True, exist_ok=True)
    cfg_path = temp_root / f"{pipeline.replace('+', '_')}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return base, cfg_path


def _is_zip_error(msg: str) -> bool:
    return "PytorchStreamReader failed reading zip archive" in msg


def _extract_superglue_missing_path(msg: str) -> Path | None:
    match = re.search(r"No such file or directory: ['\"]([^'\"]+superglue_outdoor\\.pth)['\"]", msg)
    if match:
        return Path(match.group(1))
    return None


def _purge_torch_hub_checkpoints(log=print) -> bool:
    try:
        import torch  # type: ignore

        hub_dir = Path(torch.hub.get_dir())
    except Exception as exc:  # noqa: BLE001
        log(f"[WARN] Unable to locate torch hub dir: {exc}")
        return False

    ckpt_dir = hub_dir / "checkpoints"
    if not ckpt_dir.exists():
        return False

    try:
        shutil.rmtree(ckpt_dir)
        log(f"[INFO] Cleared torch hub checkpoints: {ckpt_dir}")
        return True
    except Exception as exc:  # noqa: BLE001
        log(f"[WARN] Failed to clear torch hub checkpoints: {exc}")
        return False


def _ensure_superglue_weights(dest: Path | None, log=print) -> bool:
    if dest is None:
        try:
            import deep_image_matching  # type: ignore

            pkg_root = Path(deep_image_matching.__file__).resolve().parent
            dest = (
                pkg_root
                / "thirdparty"
                / "SuperGluePretrainedNetwork"
                / "models"
                / "weights"
                / "superglue_outdoor.pth"
            )
        except Exception as exc:  # noqa: BLE001
            log(f"[WARN] Unable to resolve SuperGlue weights path: {exc}")
            return False

    if dest.exists():
        return True

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        log(f"[INFO] Downloading SuperGlue weights to {dest}")
        urllib.request.urlretrieve(SUPERGLUE_OUTDOOR_URL, dest)
        return dest.exists()
    except (OSError, urllib.error.URLError) as exc:
        log(f"[WARN] Failed to download SuperGlue weights: {exc}")
        return False


def _maybe_recover_from_error(err: Exception, log=print) -> bool:
    msg = str(err)
    if _is_zip_error(msg):
        return _purge_torch_hub_checkpoints(log=log)
    missing = _extract_superglue_missing_path(msg)
    if missing is not None:
        return _ensure_superglue_weights(missing, log=log)
    return False


def _is_oom_error(msg: str) -> bool:
    return ("CUDA out of memory" in msg) or ("out of memory" in msg and "CUDA" in msg)


def _next_lower_quality(quality: str) -> str:
    order = ["highest", "high", "medium", "low", "lowest"]
    if quality not in order:
        return quality
    idx = order.index(quality)
    return order[min(idx + 1, len(order) - 1)]


def _reduce_max_images(max_images: int | None) -> int:
    if max_images is None:
        return 30
    if max_images > 40:
        return 40
    if max_images > 30:
        return 30
    if max_images > 20:
        return 20
    if max_images > 10:
        return 10
    return max_images


def _maybe_clear_cuda_cache(log=print) -> None:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        log(f"[WARN] Unable to clear CUDA cache: {exc}")


def _probe_pipeline(
    dir_path: Path,
    pipeline: str,
    quality: str,
    temp_root: Path,
    log=print,
) -> tuple[bool, str | None, str | None]:
    """
    Try to initialize a pipeline without running matching.
    Returns (ok, error, fallback_pipeline).
    """

    def _try_probe(p: str) -> None:
        from deep_image_matching.config import Config
        from deep_image_matching.image_matching import ImageMatcher

        safe_name = p.replace("/", "_").replace("\\", "_").replace(":", "_")
        output_dir = temp_root / "_probe" / safe_name
        base_pipeline, config_file = _resolve_pipeline(p, temp_root / "_pipeline_cfg")
        cfg = Config(
            {
                "dir": str(dir_path),
                "pipeline": base_pipeline,
                "config_file": str(config_file) if config_file else None,
                "outs": output_dir,
                "force": True,
                "quality": quality,
            }
        )
        _ = ImageMatcher(cfg)

    for attempt in range(2):
        try:
            _try_probe(pipeline)
            return True, None, None
        except Exception as exc:  # noqa: BLE001
            if attempt == 0 and _maybe_recover_from_error(exc, log=log):
                continue
            if pipeline == "sift+lightglue":
                try:
                    _try_probe("sift+kornia_matcher")
                    return True, None, "sift+kornia_matcher"
                except Exception:  # noqa: BLE001
                    pass
            return False, str(exc), None
    return False, "probe failed", None


def _iter_images(images_dir: Path) -> list[Path]:
    imgs: list[Path] = []
    for p in images_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() in IMAGE_EXTS:
            imgs.append(p)
    return sorted(imgs, key=lambda p: p.name)


def _get_image_size(path: Path) -> tuple[int, int] | None:
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as img:
            return int(img.width), int(img.height)
    except Exception:
        pass

    try:
        import cv2  # type: ignore

        img = cv2.imread(str(path))
        if img is None:
            return None
        h, w = img.shape[:2]
        return int(w), int(h)
    except Exception:
        pass

    try:
        import imageio.v3 as iio  # type: ignore

        img = iio.imread(path)
        h, w = img.shape[:2]
        return int(w), int(h)
    except Exception:
        return _get_image_size_bytes(path)


def _get_image_size_bytes(path: Path) -> tuple[int, int] | None:
    try:
        with path.open("rb") as f:
            header = f.read(24)
            if header.startswith(b"\x89PNG\r\n\x1a\n") and len(header) >= 24:
                width = int.from_bytes(header[16:20], "big")
                height = int.from_bytes(header[20:24], "big")
                return width, height
            if header[:2] != b"\xff\xd8":
                return None

            f.seek(2)
            sof_markers = {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }
            while True:
                byte = f.read(1)
                if not byte:
                    return None
                if byte != b"\xff":
                    continue
                marker = f.read(1)
                if not marker:
                    return None
                while marker == b"\xff":
                    marker = f.read(1)
                    if not marker:
                        return None
                m = marker[0]
                if m in sof_markers:
                    seg_len = f.read(2)
                    if len(seg_len) != 2:
                        return None
                    _ = f.read(1)
                    dims = f.read(4)
                    if len(dims) != 4:
                        return None
                    height = int.from_bytes(dims[0:2], "big")
                    width = int.from_bytes(dims[2:4], "big")
                    return width, height
                seg_len = f.read(2)
                if len(seg_len) != 2:
                    return None
                length = int.from_bytes(seg_len, "big")
                if length < 2:
                    return None
                f.seek(length - 2, 1)
    except Exception:
        return None


def _detect_image_sizes(images_dir: Path) -> dict[tuple[int, int], int]:
    sizes: dict[tuple[int, int], int] = {}
    for p in _iter_images(images_dir):
        size = _get_image_size(p)
        if size is None:
            continue
        sizes[size] = sizes.get(size, 0) + 1
    return sizes


def _prepare_subset_dir(
    *,
    dir_path: Path,
    temp_root: Path,
    max_images: int | None,
) -> Path:
    if max_images is None:
        return dir_path

    if max_images <= 0:
        raise ValueError("--max_images must be >= 1")

    images_dir = dir_path / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"{dir_path} 下找不到 images 目录")

    imgs = _iter_images(images_dir)
    if len(imgs) <= max_images:
        return dir_path

    subset_root = temp_root / f"_subset_{max_images}"
    if subset_root.exists():
        shutil.rmtree(subset_root)
    subset_images = subset_root / "images"
    subset_images.mkdir(parents=True, exist_ok=True)

    for src in imgs[:max_images]:
        dst = subset_images / src.name
        try:
            os.link(src, dst)  # fast, no extra disk if same volume
        except OSError:
            shutil.copy2(src, dst)

    return subset_root


def summarize_h5(feature_path: Path, match_path: Path) -> dict:
    import h5py

    summary: dict[str, object] = {}

    with h5py.File(str(feature_path), "r") as f:
        image_names = list(f.keys())
        kps_counts: list[int] = []
        for name in image_names:
            grp = f[name]
            if "keypoints" in grp:
                kps_counts.append(int(grp["keypoints"].shape[0]))
        summary["n_images"] = len(image_names)
        summary["keypoints_per_image"] = {
            "min": min(kps_counts) if kps_counts else 0,
            "max": max(kps_counts) if kps_counts else 0,
            "avg": (sum(kps_counts) / len(kps_counts)) if kps_counts else 0.0,
        }

    n_pairs = 0
    n_matches = 0
    with h5py.File(str(match_path), "r") as f:
        for key_1 in f.keys():
            grp = f[key_1]
            for key_2 in grp.keys():
                matches = grp[key_2][()]
                n_pairs += 1
                n_matches += int(matches.shape[0])
    summary["pairs"] = n_pairs
    summary["matches_total"] = n_matches
    return summary


def _get_rss_bytes() -> int | None:
    """
    Best-effort process RSS (resident set size) in bytes.
    Returns None if not supported.
    """
    # psutil is optional; prefer it if present.
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass

    # Windows fallback via ctypes.
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            GetCurrentProcess = ctypes.windll.kernel32.GetCurrentProcess
            GetProcessMemoryInfo = ctypes.windll.psapi.GetProcessMemoryInfo
            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(counters)
            ok = GetProcessMemoryInfo(GetCurrentProcess(), ctypes.byref(counters), counters.cb)
            if ok:
                return int(counters.WorkingSetSize)
        except Exception:
            return None

    return None


@dataclass
class BenchmarkMetrics:
    wall_time_s: float
    cpu_time_s: float
    rss_peak_mb: float | None = None
    rss_avg_mb: float | None = None
    cuda_peak_alloc_mb: float | None = None
    cuda_peak_reserved_mb: float | None = None


class _RssSampler:
    def __init__(self, interval_s: float) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self.samples: list[int] = []
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        def _loop() -> None:
            while not self._stop.is_set():
                rss = _get_rss_bytes()
                if rss is not None:
                    self.samples.append(int(rss))
                time.sleep(self.interval_s)

        self.thread = threading.Thread(target=_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)

    def summary_mb(self) -> tuple[float | None, float | None]:
        if not self.samples:
            return None, None
        peak = max(self.samples) / (1024 * 1024)
        avg = (sum(self.samples) / len(self.samples)) / (1024 * 1024)
        return peak, avg


def _cuda_peak_mb() -> tuple[float | None, float | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None
        alloc = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
        reserved = float(torch.cuda.max_memory_reserved() / (1024 * 1024))
        return alloc, reserved
    except Exception:
        return None, None


def _reset_cuda_peaks() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _pick_nonexistent_run_dir(base: Path, prefix: str = "run") -> Path:
    """
    Pick a non-existing run directory under `base`.
    Example: base/run_001, base/run_002, ...
    """
    for i in range(1, 10_000):
        cand = base / f"{prefix}_{i:03d}"
        if not cand.exists():
            return cand
    raise RuntimeError("Unable to pick a non-existing run directory")


def _run_cmd(cmd: list[str]) -> None:
    print(">> " + " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True)


def _find_sparse_model_dir(sparse_root: Path) -> Path:
    candidates = [sparse_root, sparse_root / "0"]
    for cand in candidates:
        if (cand / "cameras.bin").exists() and (cand / "images.bin").exists():
            return cand
    for sub in sparse_root.iterdir():
        if sub.is_dir() and (sub / "cameras.bin").exists() and (sub / "images.bin").exists():
            return sub
    raise RuntimeError(f"找不到 sparse 模型: {sparse_root}")


def _run_colmap_dense(
    *,
    colmap_bin: str,
    images_dir: Path,
    database_path: Path,
    output_dir: Path,
    overwrite: bool,
    patch_match_gpu: int | None,
    geom_verification: bool,
) -> Path:
    if not images_dir.exists():
        raise FileNotFoundError(f"找不到 images 目录: {images_dir}")
    if not database_path.exists():
        raise FileNotFoundError(f"找不到 COLMAP 数据库: {database_path}")

    sparse_root = output_dir / "sparse"
    dense_root = output_dir / "dense"
    if overwrite:
        if sparse_root.exists():
            shutil.rmtree(sparse_root)
        if dense_root.exists():
            shutil.rmtree(dense_root)
    sparse_root.mkdir(parents=True, exist_ok=True)

    if geom_verification:
        from .db_geometric_verification import geometric_verification_db

        print("[INFO] Geometric verification: Python fallback (writes two_view_geometries)")
        geometric_verification_db(str(database_path), log=print)

    _run_cmd(
        [
            colmap_bin,
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(images_dir),
            "--output_path",
            str(sparse_root),
        ]
    )

    sparse_model = _find_sparse_model_dir(sparse_root)
    dense_root.mkdir(parents=True, exist_ok=True)

    _run_cmd(
        [
            colmap_bin,
            "image_undistorter",
            "--image_path",
            str(images_dir),
            "--input_path",
            str(sparse_model),
            "--output_path",
            str(dense_root),
            "--output_type",
            "COLMAP",
        ]
    )

    cmd_patch_match = [
        colmap_bin,
        "patch_match_stereo",
        "--workspace_path",
        str(dense_root),
        "--workspace_format",
        "COLMAP",
        "--PatchMatchStereo.geom_consistency",
        "true",
    ]
    if patch_match_gpu is not None:
        cmd_patch_match += ["--PatchMatchStereo.gpu_index", str(patch_match_gpu)]
    _run_cmd(cmd_patch_match)

    fused_path = dense_root / "fused.ply"
    _run_cmd(
        [
            colmap_bin,
            "stereo_fusion",
            "--workspace_path",
            str(dense_root),
            "--workspace_format",
            "COLMAP",
            "--input_type",
            "geometric",
            "--output_path",
            str(fused_path),
        ]
    )
    return fused_path


def run_dim(
    dir_path: Path,
    pipeline: str,
    output_dir: Path,
    overwrite: bool,
    quality: str = "medium",
    *,
    single_camera: bool = True,
    camera_model: str = "simple-radial",
    max_images: int | None = None,
    print_summary: bool = False,
    temp_root: Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    """
    Run DIM feature extraction + matching and export a COLMAP database.
    Returns (feature_path, match_path, database_path, images_dir).
    """
    tmp = temp_root or (dir_path / "_tmp_dim_wrapper")
    base_pipeline, config_file = _resolve_pipeline(pipeline, tmp / "_pipeline_cfg")
    dir_path = _prepare_subset_dir(dir_path=dir_path, temp_root=tmp, max_images=max_images)

    # `output_dir` is our stable location (used by the surrounding pipeline).
    # deep-image-matching cannot reliably reuse existing outputs, and it will exit
    # if the output folder already exists and `--force` is not set.
    #
    # To make repeated runs ergonomic:
    # - if `overwrite`: remove the stable folder (fresh run)
    # - else: write DIM artifacts to a unique subfolder and only write database.db
    #   to the stable folder.
    output_dir.mkdir(parents=True, exist_ok=True)
    # Always run DIM into a non-existing subfolder; DIM's Config exits if `outs` exists.
    if overwrite:
        dim_out_dir = output_dir / "run_latest"
        if dim_out_dir.exists():
            shutil.rmtree(dim_out_dir)
        # Optional cleanup: keep old runs only when overwrite is off.
    else:
        dim_out_dir = _pick_nonexistent_run_dir(output_dir)

    images_dir = dir_path / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"{dir_path} 下找不到 images 目录")
    imgs = _iter_images(images_dir)
    if not imgs:
        sub_with_imgs: list[Path] = []
        for p in images_dir.iterdir():
            if p.is_dir() and _iter_images(p):
                sub_with_imgs.append(p)
        hint = ""
        if sub_with_imgs:
            hint = f"（发现子目录含图片：{sub_with_imgs[0].name}，请把图片移动到 images/ 下，或把该子目录内容链接/复制到 images/）"
        raise FileNotFoundError(f"{images_dir} 里没有检测到图片文件{hint}")

    from deep_image_matching.config import Config
    from deep_image_matching.image_matching import ImageMatcher

    args = {
        "dir": str(dir_path),
        "pipeline": base_pipeline,
        "outs": dim_out_dir,  # Config expects a Path-like with .exists()
        "force": False,  # we manage deletion/uniqueness ourselves
        "quality": quality,
        "config_file": str(config_file) if config_file else None,
        # Keep other options at their DIM defaults.
    }

    cfg = Config(args)
    images_dir = Path(cfg.general["image_dir"])
    matcher = ImageMatcher(cfg)
    feature_path, match_path = matcher.run()

    # Always write COLMAP DB to the stable folder so downstream steps can find it.
    db_path = output_dir / "database.db"
    if single_camera:
        sizes = _detect_image_sizes(images_dir)
        if len(sizes) > 1:
            sample = ", ".join(f"{w}x{h}({n})" for (w, h), n in list(sizes.items())[:4])
            print(
                "[WARN] Detected mixed image sizes; forcing multi_camera to avoid COLMAP undistorter errors: "
                + sample
            )
            single_camera = False
    export_matches_to_colmap_db(
        img_dir=Path(cfg.general["image_dir"]),
        feature_path=Path(feature_path),
        match_path=Path(match_path),
        database_path=db_path,
        single_camera=single_camera,
        camera_model=camera_model,
    )
    if print_summary:
        print(json.dumps(summarize_h5(Path(feature_path), Path(match_path)), ensure_ascii=False))
    return feature_path, match_path, db_path, images_dir


def _run_dim_with_repair(
    *,
    dir_path: Path,
    pipeline: str,
    output_dir: Path,
    overwrite: bool,
    quality: str,
    single_camera: bool,
    camera_model: str,
    max_images: int | None,
    print_summary: bool,
    temp_root: Path,
    fallback_pipeline: str | None = None,
    log=print,
) -> tuple[Path, Path, Path, Path, str]:
    """
    Run DIM with a small set of automatic recovery steps and optional fallback.
    Returns (features, matches, db, images_dir, pipeline_used).
    """
    attempted_repair = False
    tried_fallback = False
    oom_retries = 0
    active_pipeline = pipeline
    quality_current = quality
    max_images_current = max_images
    while True:
        try:
            feat, matches, db, images_dir = run_dim(
                dir_path,
                active_pipeline,
                output_dir,
                overwrite=overwrite,
                quality=quality_current,
                single_camera=single_camera,
                camera_model=camera_model,
                max_images=max_images_current,
                print_summary=print_summary,
                temp_root=temp_root,
            )
            return feat, matches, db, images_dir, active_pipeline
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if not attempted_repair and _maybe_recover_from_error(exc, log=log):
                attempted_repair = True
                log("[INFO] Retrying DIM after recovery step")
                continue
            if _is_oom_error(msg):
                _maybe_clear_cuda_cache(log=log)
                new_quality = _next_lower_quality(quality_current)
                new_max_images = _reduce_max_images(max_images_current)
                reduced = False
                if new_quality != quality_current:
                    log(f"[WARN] OOM: lowering quality {quality_current} -> {new_quality}")
                    quality_current = new_quality
                    reduced = True
                if max_images_current != new_max_images:
                    log(f"[WARN] OOM: limiting max_images {max_images_current} -> {new_max_images}")
                    max_images_current = new_max_images
                    reduced = True
                if reduced and oom_retries < 3:
                    oom_retries += 1
                    log("[INFO] Retrying DIM with lower-memory settings")
                    continue
            if active_pipeline == "sift+lightglue" and not tried_fallback:
                tried_fallback = True
                active_pipeline = fallback_pipeline or "sift+kornia_matcher"
                log(f"[WARN] Pipeline sift+lightglue failed; falling back to {active_pipeline}")
                continue
            raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Deep-image-matching wrapper (features + matches + COLMAP DB export)")
    parser.add_argument("--dir", required=False, help="Project directory containing an images/ folder.")
    parser.add_argument("--pipeline", default="superpoint+lightglue", help="DIM pipeline name (single run).")
    parser.add_argument(
        "--pipelines",
        default=None,
        help="Run multiple pipelines: 'all' or a comma-separated list (overrides --pipeline).",
    )
    parser.add_argument(
        "--quality",
        default="medium",
        choices=["highest", "high", "medium", "low", "lowest"],
        help="DIM image resolution preset (default: medium) to reduce memory usage.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="If set, only use the first N images from <dir>/images (copied/hardlinked into a temp subset).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for DIM artifacts (features/matches/database). Default: <dir>/dim_outputs",
    )
    parser.add_argument(
        "--multi_camera",
        action="store_true",
        help="Treat each image as its own camera (not recommended for typical single-camera UAV datasets).",
    )
    parser.add_argument(
        "--camera_model",
        default="simple-radial",
        choices=["simple-pinhole", "pinhole", "simple-radial", "opencv"],
        help="COLMAP camera model used to initialize the database (default: simple-radial).",
    )
    parser.add_argument(
        "--run_dense",
        action="store_true",
        help="Run COLMAP mapper + dense MVS after DIM and output fused point clouds.",
    )
    parser.add_argument(
        "--colmap_bin",
        default="colmap",
        help="COLMAP executable path for --run_dense (default: colmap).",
    )
    parser.add_argument(
        "--patch_match_gpu",
        type=int,
        default=None,
        help="GPU index for COLMAP patch_match_stereo (optional).",
    )
    parser.add_argument(
        "--skip_geom_verification",
        action="store_true",
        help="Skip Python geometric verification before COLMAP mapper.",
    )
    parser.add_argument("--list_pipelines", action="store_true", help="List available DIM pipelines and exit.")
    parser.add_argument(
        "--probe_pipelines",
        action="store_true",
        help="Try to initialize each pipeline (may download weights) and report OK/FAIL; no matching is run.",
    )
    parser.add_argument(
        "--print_summary",
        action="store_true",
        help="Print a JSON summary (keypoints/matches counts) after each run.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Record time/RAM/GPU usage for each run and write benchmark.{json,csv} report.",
    )
    parser.add_argument(
        "--benchmark_interval",
        type=float,
        default=0.2,
        help="Sampling interval in seconds for RSS monitoring (default: 0.2).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite outputs if they exist.")
    args = parser.parse_args(argv)

    if args.list_pipelines:
        for name in list_pipelines():
            print(name)
        return

    if not args.dir:
        raise SystemExit("--dir is required unless --list_pipelines is set")

    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"

    dir_path = Path(args.dir)
    pipelines: list[str]
    if args.pipelines:
        if args.pipelines.strip().lower() == "all":
            pipelines = list_pipelines()
        else:
            pipelines = [p.strip() for p in args.pipelines.split(",") if p.strip()]
    else:
        pipelines = [args.pipeline]

    original_pipeline_count = len(pipelines)
    single_run = (not args.probe_pipelines) and (original_pipeline_count == 1)
    # Decide output root.
    multi_run = args.probe_pipelines or (original_pipeline_count > 1)
    if not multi_run:
        base_output_dir = Path(args.output) if args.output else (dir_path / "dim_outputs")
        base_output_dir.parent.mkdir(parents=True, exist_ok=True)
    else:
        if args.probe_pipelines:
            base_output_dir = Path(args.output) if args.output else (dir_path / "dim_probes")
        else:
            base_output_dir = Path(args.output) if args.output else (dir_path / "dim_tests")
        base_output_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = (base_output_dir.parent / "_tmp") if not multi_run else (base_output_dir / "_tmp")
    tmp_root.mkdir(parents=True, exist_ok=True)

    def pick_output_dir_multi(root: Path, name: str, overwrite: bool) -> Path:
        cand = root / name
        if overwrite:
            return cand
        if not cand.exists():
            return cand
        for i in range(2, 1000):
            c2 = root / f"{name}_{i}"
            if not c2.exists():
                return c2
        raise RuntimeError("Unable to pick a non-existing output directory")

    if args.probe_pipelines:
        results: dict[str, dict] = {}
        for p in pipelines:
            ok, err, fallback = _probe_pipeline(dir_path, p, args.quality, tmp_root, log=print)
            if ok:
                entry = {"ok": True}
                if fallback:
                    entry["fallback_pipeline"] = fallback
                results[p] = entry
                if fallback:
                    print(f"[OK] {p} (fallback: {fallback})")
                else:
                    print(f"[OK] {p}")
            else:
                results[p] = {"ok": False, "error": err}
                print(f"[FAIL] {p}: {err}")
        if args.print_summary:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    fallback_map: dict[str, str] = {}
    skipped: dict[str, str] = {}
    if len(pipelines) > 1:
        filtered: list[str] = []
        for p in pipelines:
            ok, err, fallback = _probe_pipeline(dir_path, p, args.quality, tmp_root, log=print)
            if ok:
                filtered.append(p)
                if fallback:
                    fallback_map[p] = fallback
                    print(f"[WARN] {p} failed init; will fall back to {fallback}")
            else:
                skipped[p] = err or "probe failed"
                print(f"[SKIP] {p}: {err}")
        pipelines = filtered

    results: dict[str, dict] = {}
    for p, err in skipped.items():
        results[p] = {"ok": False, "skipped": True, "error": err}
    had_failure = False
    bench_rows: list[dict[str, Any]] = []
    for p in pipelines:
        safe_name = p.replace("/", "_").replace("\\", "_").replace(":", "_")
        output_dir = base_output_dir if not multi_run else pick_output_dir_multi(base_output_dir, safe_name, overwrite=args.overwrite)
        try:
            metrics: BenchmarkMetrics | None = None
            sampler: _RssSampler | None = None
            t0 = time.perf_counter()
            c0 = time.process_time()
            if args.benchmark:
                _reset_cuda_peaks()
                sampler = _RssSampler(max(0.05, float(args.benchmark_interval)))
                sampler.start()
            try:
                pipeline_to_run = fallback_map.get(p, p)
                if pipeline_to_run != p:
                    print(f"[WARN] {p} -> {pipeline_to_run}")
                feat, matches, db, images_dir, pipeline_used = _run_dim_with_repair(
                    dir_path=dir_path,
                    pipeline=pipeline_to_run,
                    output_dir=Path(output_dir),
                    overwrite=args.overwrite,
                    quality=args.quality,
                    single_camera=not args.multi_camera,
                    camera_model=args.camera_model,
                    max_images=args.max_images,
                    print_summary=args.print_summary,
                    temp_root=tmp_root,
                    fallback_pipeline=fallback_map.get(p),
                    log=print,
                )
            finally:
                if sampler is not None:
                    sampler.stop()
            if args.benchmark:
                peak_mb, avg_mb = sampler.summary_mb() if sampler is not None else (None, None)
                cuda_alloc, cuda_reserved = _cuda_peak_mb()
                metrics = BenchmarkMetrics(
                    wall_time_s=time.perf_counter() - t0,
                    cpu_time_s=time.process_time() - c0,
                    rss_peak_mb=peak_mb,
                    rss_avg_mb=avg_mb,
                    cuda_peak_alloc_mb=cuda_alloc,
                    cuda_peak_reserved_mb=cuda_reserved,
                )
            print(f"DIM_PIPELINE={p}")
            print(f"DIM_FEATURES={feat}")
            print(f"DIM_MATCHES={matches}")
            print(f"DIM_DATABASE={db}")
            fused: Path | None = None
            dense_ok = True
            dense_error: str | None = None
            if args.run_dense:
                try:
                    fused = _run_colmap_dense(
                        colmap_bin=args.colmap_bin,
                        images_dir=images_dir,
                        database_path=Path(db),
                        output_dir=Path(output_dir),
                        overwrite=args.overwrite,
                        patch_match_gpu=args.patch_match_gpu,
                        geom_verification=not args.skip_geom_verification,
                    )
                    print(f"DENSE_FUSED={fused}")
                except Exception as exc:  # noqa: BLE001
                    dense_ok = False
                    dense_error = str(exc)
                    print(f"[WARN] Dense failed for {p}: {exc}")
            entry: dict[str, Any] = {
                "ok": True,
                "features": str(feat),
                "matches": str(matches),
                "database": str(db),
            }
            if pipeline_used != p:
                entry["pipeline_used"] = pipeline_used
            if fused is not None:
                entry["fused"] = str(fused)
            if not dense_ok:
                entry["dense_ok"] = False
                entry["dense_error"] = dense_error
            if args.benchmark and metrics is not None:
                entry["benchmark"] = asdict(metrics)
                try:
                    summary = summarize_h5(Path(feat), Path(matches))
                except Exception:
                    summary = {}
                row = {
                    "pipeline": p,
                    "pipeline_used": pipeline_used,
                    "wall_time_s": metrics.wall_time_s,
                    "cpu_time_s": metrics.cpu_time_s,
                    "rss_peak_mb": metrics.rss_peak_mb,
                    "rss_avg_mb": metrics.rss_avg_mb,
                    "cuda_peak_alloc_mb": metrics.cuda_peak_alloc_mb,
                    "cuda_peak_reserved_mb": metrics.cuda_peak_reserved_mb,
                    "n_images": summary.get("n_images"),
                    "pairs": summary.get("pairs"),
                    "matches_total": summary.get("matches_total"),
                    "keypoints_avg": (summary.get("keypoints_per_image") or {}).get("avg") if summary else None,
                    "output_dir": str(output_dir),
                }
                bench_rows.append(row)
                print(f"BENCHMARK={json.dumps(row, ensure_ascii=False)}")
            results[p] = entry
        except Exception as e:  # noqa: BLE001
            if multi_run:
                print(f"[SKIP] {p}: {e}")
                results[p] = {"ok": False, "skipped": True, "error": str(e)}
            else:
                print(f"[FAIL] {p}: {e}")
                results[p] = {"ok": False, "error": str(e)}
                had_failure = True

    if multi_run:
        report_path = base_output_dir / "report.json"
        report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"REPORT={report_path}")
        if args.benchmark and bench_rows:
            bench_json = base_output_dir / "benchmark.json"
            bench_csv = base_output_dir / "benchmark.csv"
            bench_json.write_text(json.dumps(bench_rows, ensure_ascii=False, indent=2), encoding="utf-8")
            with bench_csv.open("w", encoding="utf-8", newline="") as f:
                fieldnames = list(bench_rows[0].keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(bench_rows)
            print(f"BENCHMARK_JSON={bench_json}")
            print(f"BENCHMARK_CSV={bench_csv}")

    # If this wrapper is used as a single-run step in the main pipeline, fail fast.
    if single_run and had_failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
