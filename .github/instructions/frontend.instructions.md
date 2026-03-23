# Frontend Instructions

Applies to:
- `src/uav_pipeline/gui.py`
- `src/uav_pipeline/workbench_gui.py`

Rules:
- 该 GUI 是现有 pipeline 的操作界面，不是独立产品线。
- 新的 GUI 控件必须映射到真实的 CLI / pipeline 参数，不要做只存在于界面的虚假选项。
- 优先保持参数可见、默认值清晰、日志可追踪，不为视觉变化破坏既有操作流。
- 若用户可见行为、命名或运行模式发生变化，必须同步更新 `README.md` 中的使用说明。
