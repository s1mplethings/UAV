# Repo Map

## Primary Source Areas

- `src/uav_pipeline/`
  - 主产品代码：CLI、legacy GUI、独立 workbench GUI、DIM 环境管理、DIM + COLMAP pipeline。
  - 任务涉及当前主入口、GUI 或 DIM 托管流程时，优先在这里改。

- `src/uav_3d_benchmark/`
  - legacy benchmark、几何工具、EuRoC / UseGeo / Blume 数据集导出。
  - 只有任务明确涉及数据集解析、已知位姿导出或旧流程兼容时再动。

- `scripts/`
  - 薄脚本、数据集准备脚本、以及少量辅助构建入口（如 GUI 打包脚本）。
  - 新逻辑不要堆在这里；共享逻辑应回收进 `src/`。

## Writable Support Areas

- `README.md`
  - 人类入口、快速启动、命令导航、文档导航。

- `AGENTS.md`
  - 唯一根级 agent 主合同。

- `docs/`
  - 对目标、目录、命令、流程、交付的解释展开。

- `.github/instructions/`
  - path-specific 局部规则，不承担根入口职责。

- `meta/`
  - 任务状态模板与最近一次交付报告模板。

## Caution Areas

- `pyproject.toml`
  - 包元数据和 console scripts 定义。修改入口或依赖时必须同步 `README.md` 与 `docs/03_commands_and_verification.md`。

- `requirements.txt`
  - 仅列基础依赖；不要把托管 DIM 环境依赖混进这里。

- `data/`
  - 用户数据输入区。通常不改、不删、不提交。

- `outputs/`
  - 重建产物目录。通常不改、不提交。

## Usually Do Not Touch

- `src/uav_pipeline/py39_dim_env/`
  - 托管 DIM 环境内容，不是源码。

- `/UAV/`
  - `.gitignore` 中明确忽略的本地副本 / 解压目录，不是权威源码。

- 根目录下的 `*.zip`、`*.mp4` 等大文件
  - 本地工件，不纳入执行合同。

- `__pycache__/`
  - 缓存文件，不应被手工编辑。
