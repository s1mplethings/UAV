# Delivery Contract

## Required Artifacts

- `meta/tasks/CURRENT.md`
  - 任务进行中持续更新，不要把它当长期规则文件。

- `meta/reports/LAST.md`
  - 每次交付前更新，用来沉淀最近一次结果和阻塞。

- Patch / diff
  - 所有结论都必须体现在仓库可见改动中。

## What Must Be Updated

- 当前任务状态变更时：更新 `meta/tasks/CURRENT.md`
- 任务结束时：更新 `meta/reports/LAST.md`
- 命令、入口、路由变化时：同步更新 `README.md`、`docs/03_commands_and_verification.md`，必要时更新 `AGENTS.md` 或对应 `instructions`
- 文档体系调整时：确保只有 `AGENTS.md` 继续承担根级 agent 主合同

## Verification Evidence

- 在 `meta/reports/LAST.md` 里列出实际执行过的验证命令
- 如果没有更强验证，至少记录 `python scripts/run_dim_colmap.py --help`
- 如果验证被外部依赖阻塞，写明缺的是什么，例如数据集、COLMAP 路径、conda 或 DIM 依赖

## Done

满足以下条件才算完成：

- 任务目标已经落在实际改动里
- 没有无理由扩大范围
- 最低验证基线已经执行，或已明确说明为什么无法执行
- `CURRENT.md` 与 `LAST.md` 已更新
- 若触及入口或规则，相关文档已同步

## Failure And Stop

遇到以下情况应停止继续扩展，并在 `LAST.md` 记录首个失败点：

- 缺少用户必须提供的凭证、可执行文件路径或硬约束
- 缺少关键产品决策，继续做会导致错误方向
- 需要端到端验证，但仓库内没有所需数据或外部依赖
- 发现要修改无关目录才能“顺便完成”任务
