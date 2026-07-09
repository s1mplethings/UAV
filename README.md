# UAV 3D Reconstruction Workbench

**UAV video and image-sequence 3D reconstruction workflow using deep-image-matching, COLMAP, CLI, and GUI tooling.**

This repository provides a practical reconstruction pipeline for UAV / robotics image sequences. It can extract frames from video, run deep image matching, call COLMAP, generate point clouds, preview reconstruction results, and send multi-view evidence to an OpenAI-compatible analysis endpoint.

> 中文：这是一个面向无人机视频和图像序列的三维重建工作台，核心流程是“视频/图片 → 特征匹配 → COLMAP 重建 → 点云预览 → 多视角分析”。

## Core features

- Video-to-frame extraction with sampling, blur filtering, deduplication, and frame limits.
- Deep-image-matching pipeline support, including pipeline probing and benchmark runs.
- COLMAP integration for sparse / dense reconstruction workflows.
- CLI entry point for reproducible batch processing.
- Tk GUI and standalone workbench GUI for operator-friendly usage.
- Point-cloud preview, standard view generation, and optional API-based analysis.

## Quick start

Prerequisites:

- Python 3.8+
- COLMAP installed, or use the Windows auto-detection / download path in the workbench GUI
- Optional managed DIM environment with `conda` available in PATH

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

Run the CLI:

```bash
uav-dim-colmap --dir D:/path/to/work_dir --colmap_bin "C:/Program Files/COLMAP/bin/colmap.exe"
```

Run from video input:

```bash
uav-dim-colmap --dir D:/path/to/video_run --video D:/path/to/input.mp4 --video_sample_fps 2.0 --colmap_bin "C:/Program Files/COLMAP/bin/colmap.exe"
```

For more stable keyframes:

```bash
uav-dim-colmap --dir D:/path/to/video_run --video D:/path/to/input.mp4 --video_sample_fps 1.0 --video_max_frames 24 --video_blur_threshold 2000 --video_dedupe_threshold 4.0 --video_min_gap_sec 1.0 --pipeline aliked+lightglue --dim_quality medium --colmap_bin "C:/Program Files/COLMAP/bin/colmap.exe"
```

## GUI

```bash
uav-gui
```

Standalone workbench GUI:

```bash
uav-workbench-gui
```

The workbench focuses on the full operator path:

```text
video input -> frame extraction -> point cloud generation -> preview -> standard views -> API analysis -> saved evidence
```

## Common commands

List available DIM pipelines:

```bash
uav-dim-colmap --list_dim_pipelines
```

Probe pipelines without running full matching:

```bash
uav-dim-colmap --dir D:/path/to/work_dir --probe_dim_pipelines all --test_quality lowest
```

Run pipeline comparison and benchmark output:

```bash
uav-dim-colmap --dir D:/path/to/work_dir --test_dim_pipelines all --test_quality low --benchmark --test_run_dense --overwrite
```

## Repository guide

- `src/uav_pipeline/`: main pipeline, CLI, GUI, and DIM environment logic
- `src/uav_3d_benchmark/`: EuRoC / UseGeo / Blume dataset export and legacy benchmark code
- `scripts/`: thin script entry points and dataset preprocessing utilities
- `data/`: raw data input, not committed
- `outputs/`: reconstruction output, not committed

## Status

Experimental applied reconstruction toolkit. It is intended for testing UAV / robotics reconstruction workflows and operator-facing 3D analysis prototypes.
