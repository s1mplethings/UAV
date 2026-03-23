# UAV 3D Benchmark

用于 UAV / robotics 图像序列的 deep-image-matching + COLMAP 重建仓库。当前主入口是可安装的 CLI `uav-dim-colmap`，并提供一个面向操作人员的 Tk GUI `uav-gui`；`scripts/` 里保留少量 legacy dataset / benchmark 脚本。

## Quick Start

前提：
- Python 3.8+
- 已安装 COLMAP
- 如果使用托管 DIM 环境，`conda` 需要在 PATH 中

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

CLI 运行：

```bash
uav-dim-colmap --dir D:/path/to/work_dir --colmap_bin "C:/Program Files/COLMAP/bin/colmap.exe"
```

直接从视频抽帧并生成点云：

```bash
uav-dim-colmap --dir D:/path/to/video_run --video D:/path/to/input.mp4 --video_sample_fps 2.0 --colmap_bin "C:/Program Files/COLMAP/bin/colmap.exe"
```

需要更稳的关键帧时，可以加清晰度/去重筛选：

```bash
uav-dim-colmap --dir D:/path/to/video_run --video D:/path/to/input.mp4 --video_sample_fps 1.0 --video_max_frames 24 --video_blur_threshold 2000 --video_dedupe_threshold 4.0 --video_min_gap_sec 1.0 --pipeline aliked+lightglue --dim_quality medium --colmap_bin "C:/Program Files/COLMAP/bin/colmap.exe"
```

GUI 运行：

```bash
uav-gui
```

独立工作台 GUI：

```bash
uav-workbench-gui
```

打包独立工作台 GUI（Windows）：

```bash
python scripts/build_workbench_gui.py
```

默认会产出：
- `dist/uav-workbench-gui/`
- `dist/uav-workbench-gui-windows.zip`

新的独立工作台从左到右固定是：
- 放入视频并生成点云
- 中间用鼠标可交互的点云预览检查视角，可切换 `点状 / 面状` 显示风格，并支持 `左键旋转 / 右键平移 / 滚轮缩放`
- 自动生成 4 张标准视图（`Front / Right / Top / Isometric`）
- 把这 4 张图一起发送给 OpenAI 兼容接口
- 如果点云已经生成好了，可以直接点 `分析当前点云`，不必重新跑重建
- 如果已经有点云，可以直接点 `MeshLab 打开点云`；没有 MeshLab 时，工作台会先自动检测或下载官方 Windows 版再打开
- 在最右侧显示回答，并把四视图 / 总览图 / API 原始 JSON 保存到输出目录下的 `analysis/`
- 左侧 `COLMAP` 支持 `自动检测/下载 COLMAP`：优先探测本机安装，找不到时在 Windows 下自动下载官方 COLMAP 发布包到应用内部目录并回填路径
- 工作台右侧默认只保留 API key、提示词和回答区域；模型、Base URL、detail 等高级参数默认隐藏，继续走保存值、环境变量、应用同目录 `openai.env` 或内置默认值

`uav-gui` 仍然保留，适合原来的多标签运维式操作；如果你要的是单独、完整、面向“视频 -> 点云 -> API 分析”的新界面，优先用 `uav-workbench-gui`。

其中 `work_dir` 是输出目录：
- 如果你已经有图片，目录里需要有 `images/`
- 如果你传入 `--video`，程序会先把抽帧结果写到 `work_dir/images/`
- 如果你同时给了 `--video_max_frames`，程序会先全程采样，再按时间覆盖和清晰度挑帧，而不是只截最前面的 N 帧
- DIM 和 COLMAP 输出默认也写回同一工作目录

## Common Commands

列出可用 DIM pipelines：

```bash
uav-dim-colmap --list_dim_pipelines
```

只做 pipeline 探测，不跑匹配：

```bash
uav-dim-colmap --dir D:/path/to/work_dir --probe_dim_pipelines all --test_quality lowest
```

跑 pipeline 对比测试并输出 benchmark：

```bash
uav-dim-colmap --dir D:/path/to/work_dir --test_dim_pipelines all --test_quality low --benchmark --test_run_dense --overwrite
```

用视频做 pipeline 探测 / 测试时，也可以直接加同样的 `--video`、`--video_sample_fps`、`--video_max_frames` 参数；需要更稳的关键帧时，再补 `--video_blur_threshold`、`--video_dedupe_threshold`、`--video_min_gap_sec`。

legacy / dataset-specific 示例：

```bash
python scripts/prepare_euroc.py --seq MH_01_easy --cam cam0
python scripts/run_all.py
python scripts/run_slam_stub.py
```

## Repository Guide

- `src/uav_pipeline/`: 主 pipeline、CLI、GUI、DIM 托管环境逻辑
- `src/uav_3d_benchmark/`: EuRoC / UseGeo / Blume 数据集导出与 legacy benchmark 代码
- `scripts/`: 薄脚本入口和数据预处理脚本
- `data/`: 原始数据输入，不提交
- `outputs/`: 重建产物，不提交

## Notes

- `deep-image-matching` 不是项目直接依赖；默认会创建托管的 Python 3.9 DIM 环境 `py39_dim_env`。
- Windows 打包脚本默认走 `PyInstaller onedir`，因为 Tk、OpenCV 和托管 DIM 环境在这个仓库里比 `onefile` 更稳。
- 打包后的 `uav-workbench-gui.exe` 会强制使用托管 DIM 环境；源码模式不会在冻结应用里暴露。
- Windows 打包工作台如果没找到 COLMAP，可直接在 GUI 里点 `自动检测/下载 COLMAP`；下载产物会落到应用目录下的 `_internal/runtime_tools/colmap/`。
- Windows 打包工作台如果没找到 MeshLab，可直接点 `MeshLab 打开点云`；下载产物会落到应用目录下的 `_internal/runtime_tools/meshlab/`。
- 视频输入模式会在 `work_dir/images/` 下生成抽帧结果，并在同目录写一个 `video_input.json` 供重复运行时复用。
- `video_input.json` 现在也会记录视频筛帧参数；筛帧参数变了就不会错误复用旧抽帧结果。
- `dense/fused.ply` 现在会在输出后自动做一次轻量后处理：按坐标分位数去掉少量极端离群点，并把点云包围盒中心平移到原点，便于在查看器里居中显示。
- GUI 的 OpenAI 分析会优先读取界面里的 API key；留空时回退到环境变量 `OPENAI_API_KEY`。
- 如果你希望打包后的工作台直接带默认 API 配置，可以在 `uav-workbench-gui.exe` 同目录放一个 `openai.env`，里面写 `OPENAI_API_KEY=...`，也可选填 `OPENAI_BASE_URL=...`、`OPENAI_MODEL=...`；GUI 会读取它，但不会把 key 写回配置文件。
- 如果你使用兼容 OpenAI 协议的第三方网关，也可以在 GUI 里填写自定义 `API Base URL`；默认仍是官方 `https://api.openai.com/v1`。
- 如果 API 调用失败，工作台现在会把失败原因同时写进右侧回答区，而不只是弹一个错误框。
- GUI 工作台会额外在 `analysis/api_views/` 下生成四张单独图片，发送给 API 的就是这四张，而不是只发一张拼图。
- `面状` 只是显示层的补面渲染，让稀疏点云看起来更接近表面；它不会改动真实 `.ply`，也不会生成真正的 mesh 文件。
- GUI 不会把输入过的 API key 保存到本地配置文件。
- 点云截图使用仓库内置的轻量渲染，不依赖 Open3D 或 matplotlib。
- 对应的原始 COLMAP 输出会保留为 `dense/fused_raw.ply`，处理参数记录在 `dense/fused_postprocess.json`。
- 单相机 UAV 数据通常保持默认 `single_camera` 行为即可；多相机或变焦数据再考虑 `--dim_multi_camera`。
- 如果报错找不到 `colmap`，请传入 `--colmap_bin` 的完整路径，或在工作台里使用 `自动检测/下载 COLMAP`。
- GUI 配置会写到用户目录下的 `.uav_pipeline_config.json`，不会写回仓库。

## Documentation

- Automation contract: `AGENTS.md`
- Project north star: `docs/01_north_star.md`
- Repo map: `docs/02_repo_map.md`
- Commands and verification: `docs/03_commands_and_verification.md`
- Execution flow: `docs/04_execution_flow.md`
- Delivery contract: `docs/05_delivery_contract.md`
