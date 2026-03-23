# Execution Flow

本文件是对根 `AGENTS.md` 中单一流程的展开说明，不替代根合同。

## 1. Bind

- 先把任务收敛成一个主题，例如“改 CLI 参数行为”或“重写文档体系”。
- 先确认主要作用域，再写入 `meta/tasks/CURRENT.md`。
- 如果中途发现任务已经跨主题，先 rebinding，再继续改。

## 2. Read

- 先读 `AGENTS.md`。
- 再读与当前任务直接相关的源码、脚本、文档。
- 根据路径加载局部规则：
  - 文档类改动看 `.github/instructions/docs.instructions.md`
  - pipeline / 入口类改动看 `.github/instructions/workflow.instructions.md`
  - GUI 类改动看 `.github/instructions/frontend.instructions.md`

## 3. Analyze

- 判断主入口是否还是 `uav-dim-colmap`；不要让 legacy 脚本升级成平行主入口。
- 识别最小改动面，避免顺手扩散到 `data/`、`outputs/`、`py39_dim_env/`、忽略副本目录。
- 先挑一个真实可跑的验证命令，再开始改。

## 4. Change

- 共享逻辑回 `src/`，脚本继续保持薄包装。
- 根 `AGENTS.md` 只留硬规则，不在这里重复长篇解释。
- `README.md` 只给人类用，不写 agent 流程细则。
- `docs/` 的每个文件只承担自己的说明职责，出现重复时把内容向下沉到更专门的文件。

## 5. Verify / Close

- 至少运行 `python scripts/run_dim_colmap.py --help`。
- 若改动涉及被 git 跟踪的 Python 源文件、安装或 wrapper，再补 `docs/03_commands_and_verification.md` 里的 git-tracked Python syntax check 或 `python -m pip install -e .`。
- 关闭任务前更新 `meta/reports/LAST.md`，明确：
  - 做了什么
  - 改了哪些文件
  - 跑了哪些验证
  - 结果是完成、部分完成还是阻塞
  - 第一个阻塞失败是什么
