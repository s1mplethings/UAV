uav_3d_benchmark
=================

Minimal testbed to run COLMAP with known poses on EuRoC MAV and UseGeo UAV strips.

Layout
------
- `uav_3d_benchmark/` Python package with dataset parsers and the COLMAP pipeline wrapper.
- `data/` Put raw datasets here:
  - `data/euroc/<sequence>/mav0/...` (e.g., `MH_01_easy`)
  - `data/usegeo/strip1/...` (update paths in `UseGeoConfig` if your layout differs)
- `outputs/` Generated sparse/dense reconstructions.
- `scripts/` Convenience entrypoints.

Quickstart
----------
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Prepare data folders as above, ensure `colmap` is on PATH.
python scripts/run_all.py
```

Key pieces
----------
- EuRoC: uses `cam0` intrinsics/extrinsics and ground-truth poses to emit COLMAP `cameras.txt`/`images.txt`.
- UseGeo: parses provided intrinsics and omega/phi/kappa camera poses, converts to COLMAP format.
- `colmap_pipeline.py`: shared SfM+MVS steps (feature extraction, matching, triangulation, undistortion, stereo, fusion).

Notes
-----
- Geometry helpers live in `uav_3d_benchmark/geometry.py`.
- If your dataset file names differ (pose/intrinsics for UseGeo), adjust the paths/parsers in `datasets/usegeo.py`.
- `eval/metrics.py` is a stub for future LiDAR/depth evaluation.
