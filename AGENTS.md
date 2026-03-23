# UAV 3D Benchmark Agent Contract

## 1. Purpose
- 本仓库用于把 UAV / robotics 图像工作目录或视频输入接入 deep-image-matching + COLMAP，输出 sparse / dense 重建结果。
- 当前主产品面是 `uav_pipeline` 的 CLI + Tk GUI；`src/uav_3d_benchmark/` 和 `scripts/` 保留数据集导出与 legacy benchmark 支持。
- Agent 在这里默认做最小改动、单主题 patch、可验证收口，不扩散到无关代码、数据或产物目录。

## 2. Single Entry
- Canonical execution entry: `uav-dim-colmap` from `pyproject.toml`; repo 内等价源码入口是 `src/uav_pipeline/cli.py`。
- Canonical verification entry: `python scripts/run_dim_colmap.py --help`。
- 只有任务明确要求兼容 legacy flow 时，才使用 `scripts/run_dim_colmap.py`、`scripts/run_all.py` 或 `scripts/prepare_*.py`。
- `README.md` 只做人类入口；agent 进入仓库后的第一入口是本文件。

## 3. Single Flow
1. Bind
- 锁定一个任务、一个作用域、一个主要验收目标。
- 开始做实质修改前，先把当前任务状态写入 `meta/tasks/CURRENT.md`。

2. Read
- 先读本文件，再只读完成当前任务所需的文件。
- 如果任务会改文档、GUI、执行流，先读对应的 `.github/instructions/*.instructions.md`。

3. Analyze
- 优先复用现有 CLI、pipeline、脚本和配置布局，不新造平行入口。
- 先确定最小改动面、真实依赖边界、以及足以证明完成的最小验证。

4. Change
- 一个 patch 只做一个主题。
- 共享逻辑放 `src/`，薄包装放 `scripts/`，解释文档放 `docs/`。
- 默认不要编辑 `data/`、`outputs/`、`/UAV/`、媒体/压缩产物、`src/uav_pipeline/py39_dim_env/` 这类输入、输出、忽略副本或托管环境内容。

5. Verify/Close
- 最低要求执行 `python scripts/run_dim_colmap.py --help`。
- 如果改动触及被 git 跟踪的 Python 源文件、安装、GUI 或脚本，再补最窄的针对性验证。
- 结束前把结果、验证、改动文件和首个阻塞失败写入 `meta/reports/LAST.md`。

## 4. Allowed Questions
- 缺少凭证、账号、密钥、许可证、可执行文件路径，仓库本身无法推断。
- 存在会改变主流程或界面形态的互斥产品决策，需要用户拍板。
- 缺少关键硬约束，继续推进会产生不安全或高概率错误的结果。
- 除此之外默认先推进，并在交付物中记录所做假设。

## 5. Required Outputs
- `meta/tasks/CURRENT.md`：任务进行中持续更新 task / scope / constraints / current step / next action / status。
- `meta/reports/LAST.md`：关闭任务前更新 goal / changed files / verification / result / first failure / next recommended step。
- 真实 patch / diff：所有结论都必须落在仓库可见改动中，不做隐藏交付。
- 若命令、入口、路由或交付规则发生变化，相关 `README.md` / `docs/` / `.github/instructions/` 必须同步。

## 6. Non-Negotiable Rules
- One topic per patch。
- 没有 rebinding，就不要扩大任务范围。
- 不得跳过 canonical verification entry；需要时再补更强验证。
- 不得修改无关文件、生成产物或用户数据。
- 优先使用现有脚本、instructions、包入口和配置约定，不新造流程。
- `AGENTS.md` 必须保持薄主合同；长解释下沉到 `docs/`。
- `README.md` 必须保持人类入口；不要把它写成第二份 agent 合同。
- 版本库根目录是唯一权威源码；忽略的本地副本（如 `/UAV/`）永远不是权威来源。

## 7. Routing
- Path-specific rules: `.github/instructions/`。
- Reusable repo workflows and executable entrypoints: `pyproject.toml`、`src/uav_pipeline/`、`src/uav_3d_benchmark/`、`scripts/`。
- Human-oriented explanations: `README.md` 和 `docs/01_north_star.md` 到 `docs/05_delivery_contract.md`。
- Automation gates and verification: `docs/03_commands_and_verification.md`。
- Task state and closeout artifacts: `meta/tasks/CURRENT.md` 和 `meta/reports/LAST.md`。
- 若出现规则重复，保留本文件中的短硬规则，把解释迁移到匹配的 `docs/` 或 `.github/instructions/` 文件。
