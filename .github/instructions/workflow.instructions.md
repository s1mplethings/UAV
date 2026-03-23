# Workflow Instructions

Applies to:
- `pyproject.toml`
- `scripts/**`
- `src/uav_pipeline/**`
- `src/uav_3d_benchmark/**`

Rules:
- 把 `uav-dim-colmap` 视为默认主入口；不要再引入新的根级平行执行流。
- 共享逻辑进 `src/`，`scripts/` 继续保持薄包装或数据集特化入口。
- 保持工作目录约定稳定：用户提供一个 `work_dir`；它要么已经含 `images/`，要么由 `--video` 先生成 `images/`，输出回写到该目录或 `outputs/`。
- 不要手工维护 `src/uav_pipeline/py39_dim_env/`；它是托管环境内容，不是源代码。
- 若修改了入口、参数、安装方式或工作流约定，必须同步更新 `README.md` 与 `docs/03_commands_and_verification.md`。
