"""
Minimal wrapper to run deep-image-matching as a CLI-friendly script.

This uses the deep_image_matching library directly (since the package ships
without a __main__) and exports a COLMAP database from the produced features
and matches. It does *not* run COLMAP mapper; the caller can do that after this
script finishes (our pipeline does it in Python).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deep_image_matching import Config, ImageMatcher
from deep_image_matching.io.h5_to_db import export_to_colmap


def run_dim(dir_path: Path, pipeline: str, output_dir: Path, overwrite: bool) -> tuple[Path, Path, Path]:
    """
    Run DIM feature extraction + matching and export a COLMAP database.
    Returns (feature_path, match_path, database_path).
    """
    args = {
        "dir": str(dir_path),
        "pipeline": pipeline,
        "outs": str(output_dir),
        "force": overwrite,
        # Keep other options at their DIM defaults.
    }

    cfg = Config(args)
    matcher = ImageMatcher(cfg)
    feature_path, match_path = matcher.run()

    db_path = output_dir / "database.db"
    export_to_colmap(cfg.general["image_dir"], feature_path, match_path, database_path=str(db_path))
    return feature_path, match_path, db_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Deep-image-matching wrapper (features + matches + COLMAP DB export)")
    parser.add_argument("--dir", required=True, help="Project directory containing an images/ folder.")
    parser.add_argument("--pipeline", default="superpoint+lightglue", help="DIM pipeline name.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory for DIM artifacts (features/matches/database). Default: <dir>/dim_outputs",
    )
    parser.add_argument("--overwrite", action="store_true", help="Force overwrite outputs if they exist.")
    args = parser.parse_args(argv)

    dir_path = Path(args.dir)
    output_dir = Path(args.output) if args.output else dir_path / "dim_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    feat, matches, db = run_dim(dir_path, args.pipeline, output_dir, overwrite=args.overwrite)
    print(f"DIM_FEATURES={feat}")
    print(f"DIM_MATCHES={matches}")
    print(f"DIM_DATABASE={db}")


if __name__ == "__main__":
    main()
