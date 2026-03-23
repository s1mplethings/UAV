# North Star

## Goal

- 让一个包含 `images/` 的工作目录，或一个可抽帧的视频输入，可以通过现有 CLI 或 GUI 接入 deep-image-matching + COLMAP，稳定产出 sparse / dense 重建结果。
- 保留 EuRoC、UseGeo、Blume 等数据集的导出和 legacy benchmark 能力，但不让它们抢主入口。
- 让 agent 与人类都能从少量文件快速判断入口、验证方式和交付物。

## Non-Goals

- 不是通用的视觉算法实验平台，也不是新的多入口脚手架。
- 不负责管理原始数据、重建结果、压缩包、视频或托管虚拟环境内容。
- 不把 `README.md`、`AGENTS.md`、`docs/`、`.github/instructions/` 写成多头权威体系。

## Core Principles

- Single front door: 主执行入口是 `uav-dim-colmap`，主验收入口是 `python scripts/run_dim_colmap.py --help`。
- Thin wrappers: 共享逻辑进 `src/`，`scripts/` 只做薄包装或数据集特化入口。
- Minimal change: 一个任务只收敛一个主题，不顺手重构不相关路径。
- Real verification: 只写仓库里真实可跑的命令，不虚构 CI、测试或 lint 体系。
- Clear authority split: 根 `AGENTS.md` 只放硬规则；`README.md` 只做人类入口；`docs/` 负责展开说明；`.github/instructions/` 只放局部规则。
