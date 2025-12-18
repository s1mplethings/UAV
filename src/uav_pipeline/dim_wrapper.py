"""
Minimal wrapper to run deep-image-matching as a CLI-friendly script.

This uses the deep_image_matching library directly (since the package ships
without a __main__) and exports a COLMAP database from the produced features
and matches. It does *not* run COLMAP mapper; the caller can do that after this
script finishes (our pipeline does it in Python).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

PIPELINE_ALIASES: dict[str, dict] = {
    # Commonly used combo but not shipped as a built-in DIM pipeline name.
    "sift+lightglue": {
        "base_pipeline": "sift+kornia_matcher",
        "config": {"matcher": {"name": "lightglue"}},
    },
}


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


def _iter_images(images_dir: Path) -> list[Path]:
    imgs: list[Path] = []
    for p in images_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() in IMAGE_EXTS:
            imgs.append(p)
    return sorted(imgs, key=lambda p: p.name)


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
) -> tuple[Path, Path, Path]:
    """
    Run DIM feature extraction + matching and export a COLMAP database.
    Returns (feature_path, match_path, database_path).
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
    dim_out_dir = output_dir
    if overwrite:
        # Full reset of DIM artifacts + exported database in this folder.
        for child in list(output_dir.iterdir()):
            if child.is_dir():
                shutil.rmtree(child)
            else:
                try:
                    child.unlink()
                except FileNotFoundError:
                    pass
    else:
        # Keep old artifacts; put new DIM outputs into a run-specific folder.
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
    matcher = ImageMatcher(cfg)
    feature_path, match_path = matcher.run()

    # Always write COLMAP DB to the stable folder so downstream steps can find it.
    db_path = output_dir / "database.db"
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
    return feature_path, match_path, db_path


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
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite outputs if they exist.")
    args = parser.parse_args(argv)

    if args.list_pipelines:
        for name in list_pipelines():
            print(name)
        return

    if not args.dir:
        raise SystemExit("--dir is required unless --list_pipelines is set")

    dir_path = Path(args.dir)
    pipelines: list[str]
    if args.pipelines:
        if args.pipelines.strip().lower() == "all":
            pipelines = list_pipelines()
        else:
            pipelines = [p.strip() for p in args.pipelines.split(",") if p.strip()]
    else:
        pipelines = [args.pipeline]

    # Decide output root.
    multi_run = args.probe_pipelines or (len(pipelines) > 1)
    if not multi_run:
        base_output_dir = Path(args.output) if args.output else (dir_path / "dim_outputs")
        base_output_dir.parent.mkdir(parents=True, exist_ok=True)
    else:
        if args.probe_pipelines:
            base_output_dir = Path(args.output) if args.output else (dir_path / "dim_probes")
        else:
            base_output_dir = Path(args.output) if args.output else (dir_path / "dim_tests")
        base_output_dir.mkdir(parents=True, exist_ok=True)

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
            try:
                from deep_image_matching.config import Config
                from deep_image_matching.image_matching import ImageMatcher

                safe_name = p.replace("/", "_").replace("\\", "_").replace(":", "_")
                output_dir = pick_output_dir_multi(base_output_dir, safe_name, overwrite=True)
                tmp_root = base_output_dir / "_tmp"
                base_pipeline, config_file = _resolve_pipeline(p, tmp_root / "_pipeline_cfg")
                cfg = Config(
                    {
                        "dir": str(dir_path),
                        "pipeline": base_pipeline,
                        "config_file": str(config_file) if config_file else None,
                        "outs": output_dir,
                        "force": True,
                        "quality": args.quality,
                    }
                )
                _ = ImageMatcher(cfg)
                results[p] = {"ok": True}
                print(f"[OK] {p}")
            except Exception as e:  # noqa: BLE001
                results[p] = {"ok": False, "error": str(e)}
                print(f"[FAIL] {p}: {e}")
        if args.print_summary:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    results: dict[str, dict] = {}
    had_failure = False
    for p in pipelines:
        safe_name = p.replace("/", "_").replace("\\", "_").replace(":", "_")
        output_dir = base_output_dir if not multi_run else pick_output_dir_multi(base_output_dir, safe_name, overwrite=args.overwrite)
        try:
            feat, matches, db = run_dim(
                dir_path,
                p,
                output_dir,
                overwrite=args.overwrite,
                quality=args.quality,
                single_camera=not args.multi_camera,
                camera_model=args.camera_model,
                max_images=args.max_images,
                print_summary=args.print_summary,
                temp_root=(base_output_dir.parent / "_tmp") if not multi_run else (base_output_dir / "_tmp"),
            )
            print(f"DIM_PIPELINE={p}")
            print(f"DIM_FEATURES={feat}")
            print(f"DIM_MATCHES={matches}")
            print(f"DIM_DATABASE={db}")
            results[p] = {"ok": True, "features": str(feat), "matches": str(matches), "database": str(db)}
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {p}: {e}")
            results[p] = {"ok": False, "error": str(e)}
            had_failure = True

    if len(pipelines) > 1:
        report_path = base_output_dir / "report.json"
        report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"REPORT={report_path}")

    # If this wrapper is used as a single-run step in the main pipeline, fail fast.
    if len(pipelines) == 1 and had_failure and not args.probe_pipelines:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
