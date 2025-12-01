uav_3d_benchmark
=================

Minimal testbed to run COLMAP with known poses on EuRoC MAV and UseGeo UAV strips. Now packaged as a `src/` layout with CLI + Tk GUI for the DIM + COLMAP pipeline.

Layout
------
- `src/uav_3d_benchmark/` Python package with dataset parsers and the COLMAP pipeline wrapper.
- `src/uav_pipeline/` Reusable DIM+COLMAP pipeline (CLI + GUI).
- `data/` Put raw datasets here:
  - `data/euroc/<sequence>/mav0/...` (e.g., `MH_01_easy`)
  - `data/usegeo/strip1/...` (update paths in `UseGeoConfig` if your layout differs)
- `outputs/` Generated sparse/dense reconstructions.
- `scripts/` Convenience entrypoints (COLMAP + SLAM stub).

Quickstart
----------
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .  # installs uav_3d_benchmark + uav_pipeline (CLI + GUI) and deps

# DIM + COLMAP (CLI):
uav-dim-colmap --dir D:/path/to/work_dir --colmap_bin "C:/Program Files/COLMAP/bin/colmap.exe" --gpu 0 --patch_match_gpu 0
# or open the GUI:
uav-gui

# Legacy examples:
python scripts/run_all.py
python scripts/run_slam_stub.py
```

Key pieces
----------
- EuRoC: uses `cam0` intrinsics/extrinsics and ground-truth poses to emit COLMAP `cameras.txt`/`images.txt`.
- UseGeo: parses provided intrinsics and omega/phi/kappa camera poses, converts to COLMAP format.
- `colmap_pipeline.py`: shared SfM+MVS steps (feature extraction, matching, triangulation, undistortion, stereo, fusion).
- `uav_pipeline.pipeline`: reusable DIM + COLMAP dense pipeline with a logger hook for GUIs.

Notes
-----
- Geometry helpers live in `uav_3d_benchmark/geometry.py`.
- If your dataset file names differ (pose/intrinsics for UseGeo), adjust the paths/parsers in `datasets/usegeo.py`.
- `eval/metrics.py` is a stub for future LiDAR/depth evaluation.
