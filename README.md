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
uav-dim-colmap --dir D:/path/to/work_dir --colmap_bin "C:/Program Files/COLMAP/bin/colmap.exe" --gpu 0 --patch_match_gpu 0 --dim_quality high
# or open the GUI:
uav-gui

# 查看 / 测试 DIM pipelines（模型组合）
uav-dim-colmap --list_dim_pipelines
uav-dim-colmap --dir D:/path/to/work_dir --probe_dim_pipelines all --test_quality lowest
uav-dim-colmap --dir D:/path/to/work_dir --test_dim_pipelines all --test_quality low --overwrite

# GUI 里也可以直接做同样的测试：
# - “测试”页：填写 pipelines（all/逗号分隔）、N、quality，然后点 “Probe pipelines / 跑 smoke test”。
# - “运行”页：用“模式”下拉选择 “全流程 / 仅 DIM / 仅 Dense”，再点 “开始运行”。

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
- deep-image-matching is intentionally *not* a project dependency. The pipeline will auto-create a Python 3.9 conda env (`py39_dim_env`) beside the executable and install DIM there. Make sure `conda` is on PATH; disable the managed env with `--no_dim_env` or in the GUI checkbox if you want to run DIM in the current Python environment.
- DIM→COLMAP 的稀疏重建质量通常取决于两点：是否用 `single_camera`（默认开启，适合大多数单相机 UAV 数据）以及是否跑 `colmap geometric_verification`（默认开启）。如果你的数据确实是多相机/变焦，可以加 `--dim_multi_camera`。

Troubleshooting
---------------
- 报错 `Numpy is not available` / 提示 NumPy 2.x 不兼容 torch：这是 managed DIM env 里 numpy 版本过新导致。可以在 env 内强制降级：
  - Windows 示例：`<work_dir 或仓库>/src/uav_pipeline/py39_dim_env/python.exe -m pip install --upgrade --force-reinstall "numpy<2"`

Matching pipelines 选型（DIM pipeline 名称对照）
---------------------------------------------
1) 稀疏关键点匹配（传统 SfM，通常最稳）
- `sift+kornia_matcher`: SIFT + NN 类匹配（更接近传统路线）
- `sift+lightglue`: SIFT + LightGlue（保留 SIFT 的“硬”特性 + 更强匹配器；此项目通过 wrapper alias 支持）

2) 学习型关键点+描述子 + 学习型匹配器（工程常用主力）
- `superpoint+lightglue` / `superpoint+lightglue_fast`
- `disk+lightglue`
- `aliked+lightglue`（很多航拍场景会比 SuperPoint 更稳一点）

3) Detector-free / 半稠密（困难图对的“强力补位”）
- `loftr` / `se2loftr`

4) 稠密匹配（“终极兜底”，更吃算力）
- `roma`
