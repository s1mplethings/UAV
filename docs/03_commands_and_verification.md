# Commands And Verification

## Assumptions

- Python 3.8+
- 已安装 COLMAP
- 若使用托管 DIM 环境，`conda` 需要在 PATH 中

## Setup / Install

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

安装仓库：

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

## Canonical Run

CLI：

```bash
uav-dim-colmap --dir D:/path/to/work_dir --colmap_bin "C:/Program Files/COLMAP/bin/colmap.exe"
```

视频输入：

```bash
uav-dim-colmap --dir D:/path/to/video_run --video D:/path/to/input.mp4 --video_sample_fps 2.0 --colmap_bin "C:/Program Files/COLMAP/bin/colmap.exe"
```

需要更稳的关键帧时：

```bash
uav-dim-colmap --dir D:/path/to/video_run --video D:/path/to/input.mp4 --video_sample_fps 1.0 --video_max_frames 24 --video_blur_threshold 2000 --video_dedupe_threshold 4.0 --video_min_gap_sec 1.0 --pipeline aliked+lightglue --dim_quality medium --colmap_bin "C:/Program Files/COLMAP/bin/colmap.exe"
```

GUI：

```bash
uav-gui
```

独立工作台 GUI：

```bash
uav-workbench-gui
```

Windows 打包独立工作台 GUI：

```bash
python scripts/build_workbench_gui.py
```

说明：

- `uav-gui` 保留原来的多标签操作界面
- `uav-workbench-gui` 是单独的完整新界面，固定按“视频 / 点云输入 -> 可转动点云预览 -> 四张 API 视图 -> API 输出”组织
- 打包的 canonical 方式是 `PyInstaller onedir`；产物默认写到 `dist/uav-workbench-gui/`，并额外生成 `dist/uav-workbench-gui-windows.zip`
- 打包后的工作台会强制使用托管 DIM 环境，不再暴露源码模式
- Windows 工作台支持 `自动检测/下载 COLMAP`：先探测本机安装，找不到时再下载官方发布包到应用内部目录并自动回填路径

工作目录约定：

- `work_dir` 是输出根目录
- 如果没有 `--video`，则 `work_dir` 至少包含 `images/`
- 如果传入 `--video`，程序会先生成 `work_dir/images/`，并写入 `video_input.json` 记录抽帧配置
- 如果同时给了 `--video_max_frames`，程序会先全程采样，再按时间覆盖和清晰度挑帧，而不是只取最前面的 N 帧
- 可选筛帧参数是 `--video_blur_threshold`、`--video_dedupe_threshold`、`--video_min_gap_sec`
- DIM、sparse、dense 等输出默认写回该目录
- `dense/fused.ply` 默认会被轻量后处理成“去少量极端离群点 + 包围盒居中到原点”的版本；原始结果保留为 `dense/fused_raw.ply`
- legacy dataset scripts 仅在任务明确针对数据集导出时使用

## Legacy / Secondary Commands

EuRoC 数据集导出 + COLMAP：

```bash
python scripts/prepare_euroc.py --seq MH_01_easy --cam cam0
```

批量 legacy 示例：

```bash
python scripts/run_all.py
python scripts/run_slam_stub.py
```

## Test / Lint / Typecheck Reality

当前仓库未提交以下入口：

- `tests/`
- `.github/workflows/`
- `pytest` / `unittest` 测试套件
- `ruff` / `flake8` / `black` / `mypy` / `pyright` 配置
- `Makefile` / `tox` / `nox`

不要在文档里虚构这些命令。

## Canonical Verification

主验收入口：

```bash
python scripts/run_dim_colmap.py --help
```

如果改动触及被 git 跟踪的 Python 源文件，再补这个语法检查：

```bash
python -c "import py_compile, subprocess; files=[p for p in subprocess.check_output(['git','ls-files','src','scripts']).decode().splitlines() if p.endswith('.py')]; [py_compile.compile(f, doraise=True) for f in files]"
```

改动触及安装入口或打包方式时，再补这些真实命令中的最窄一项：

```bash
python -m pip install -e .
```

```bash
python scripts/build_workbench_gui.py --no-zip
```

涉及真实重建链路时，只有在具备数据集、COLMAP、以及所需 DIM 依赖时才跑端到端命令；否则应在交付报告中明确记录阻塞条件。

## Platform Notes

- Windows 下 `--colmap_bin` 常常需要完整 `.exe` 路径。
- GUI 会自动尝试探测常见 Windows 安装位置；工作台在 Windows 打包版下还可以自动下载官方 COLMAP 到 `_internal/runtime_tools/colmap/`；CLI 不会替用户修正路径。
- 托管 DIM 环境会写入 `src/uav_pipeline/py39_dim_env/` 或相邻工作目录，不应当作源码维护。
