"""Tkinter GUI for the deep-image-matching + COLMAP pipeline.

This GUI is meant to be "ops friendly":
- Common parameters are grouped and labeled.
- Key choices are dropdowns instead of free-text when possible.
- Supports multiple run modes (full pipeline / DIM only / dense only / tests).
"""

from __future__ import annotations

import os
import queue
import threading
import shutil
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from .dim_env import DeepImageMatchingEnv
from .openai_analysis import (
    DEFAULT_OPENAI_BASE_URL,
    OPENAI_API_KEY_ENV,
    OPENAI_BASE_URL_ENV,
    analyze_images_with_openai,
)
from .pipeline import PipelineConfig, run_cmd, run_colmap_mvs, run_dim as run_dim_step, run_pipeline
from .point_cloud_snapshot import (
    DEFAULT_POINT_CLOUD_API_VIEWS,
    PointCloudRenderData,
    prepare_point_cloud_render_data,
    render_point_cloud_snapshot,
    render_point_cloud_view,
    render_point_cloud_view_set,
)
from .user_config import load_section, update_section
from .video_input import prepare_work_dir_from_video

DIM_QUALITY_OPTIONS = ("highest", "high", "medium", "low", "lowest")
CAMERA_MODEL_OPTIONS = ("simple-radial", "simple-pinhole", "pinhole", "opencv")
OPENAI_IMAGE_DETAIL_OPTIONS = ("low", "high", "auto")
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_ANALYSIS_PROMPT = (
    "请用中文综合分析这组点云四视图：\n"
    "1. 这更像什么场景或结构；\n"
    "2. 重建是否完整，哪里缺失明显；\n"
    "3. 噪点或伪影主要集中在哪；\n"
    "4. 如果要继续采集，请给出 3 条简短建议。"
)

# Curated defaults for UAV SfM.
PIPELINE_PRESETS = (
    "superpoint+lightglue",
    "superpoint+lightglue_fast",
    "aliked+lightglue",
    "disk+lightglue",
    "sift+kornia_matcher",
    "sift+lightglue",
    "loftr",
    "se2loftr",
    "roma",
)

TEST_PIPELINE_RECOMMENDED = "sift+kornia_matcher,sift+lightglue,aliked+lightglue,superpoint+lightglue,loftr,se2loftr,roma"

MODE_FULL = "全流程：DIM → Sparse → Dense"
MODE_DIM_ONLY = "仅 DIM：特征/匹配 → Sparse"
MODE_DENSE_ONLY = "仅 Dense：使用已有 Sparse"
WORKBENCH_PREVIEW_SIZE = 520
WORKBENCH_API_VIEW_SIZE = 240

def _detect_colmap_bin() -> str:
    """
    Best-effort COLMAP binary detection for Windows-friendly UX.
    Falls back to 'colmap' (requires PATH).
    """
    found = shutil.which("colmap")
    if found:
        return found
    if os.name == "nt":
        candidates = [
            r"C:\Program Files\COLMAP\bin\colmap.exe",
            r"C:\Program Files (x86)\COLMAP\bin\colmap.exe",
        ]
        for c in candidates:
            if Path(c).exists():
                return c
    return "colmap"


class ScrollableFrame(ttk.Frame):
    """
    A vertically scrollable frame (works well for long forms).

    Usage:
        sf = ScrollableFrame(parent)
        sf.pack(fill="both", expand=True)
        # Put widgets into sf.content
    """

    def __init__(self, parent: tk.Misc, *args, **kwargs) -> None:
        super().__init__(parent, *args, **kwargs)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vscroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vscroll.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.vscroll.pack(side="right", fill="y")

        self.content = ttk.Frame(self.canvas)
        self._window_id = self.canvas.create_window((0, 0), window=self.content, anchor="nw")

        self.content.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Mouse wheel scrolling (Windows/macOS) + Linux button scroll.
        self.canvas.bind("<Enter>", lambda _e: self._bind_mousewheel())
        self.canvas.bind("<Leave>", lambda _e: self._unbind_mousewheel())

    def _on_frame_configure(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self._window_id, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        # Windows: event.delta is multiples of 120. macOS can be smaller.
        delta = int(getattr(event, "delta", 0))
        if delta:
            self.canvas.yview_scroll(int(-1 * (delta / 120)), "units")

    def _on_linux_scroll_up(self, _event: tk.Event) -> None:
        self.canvas.yview_scroll(-1, "units")

    def _on_linux_scroll_down(self, _event: tk.Event) -> None:
        self.canvas.yview_scroll(1, "units")

    def _bind_mousewheel(self) -> None:
        # Bind to toplevel so wheel works even if focus is on an Entry/Combobox.
        toplevel = self.winfo_toplevel()
        toplevel.bind_all("<MouseWheel>", self._on_mousewheel, add="+")
        toplevel.bind_all("<Button-4>", self._on_linux_scroll_up, add="+")
        toplevel.bind_all("<Button-5>", self._on_linux_scroll_down, add="+")

    def _unbind_mousewheel(self) -> None:
        toplevel = self.winfo_toplevel()
        try:
            toplevel.unbind_all("<MouseWheel>")
            toplevel.unbind_all("<Button-4>")
            toplevel.unbind_all("<Button-5>")
        except Exception:
            pass


class PipelineGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("UAV DIM + COLMAP Pipeline")
        root.minsize(1360, 780)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.running = False
        self.preview_photo: tk.PhotoImage | None = None
        self.api_view_photos: dict[str, tk.PhotoImage] = {}
        self.analysis_api_view_paths: list[str] = []
        self._render_data_cache: PointCloudRenderData | None = None
        self._render_data_cache_path: str | None = None
        self._saved_config = load_section("gui")

        self._style()
        self._build_layout()
        self._build_log()
        self._apply_saved_config()
        self._poll_logs()

    def _style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 10, "bold"))

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        self.outer = outer

        header = ttk.Frame(outer)
        header.pack(fill=tk.X)
        ttk.Label(header, text="UAV DIM + COLMAP Pipeline", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Button(header, text="清空日志", command=self.clear_log).pack(side=tk.RIGHT)

        self.notebook = ttk.Notebook(outer)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=(10, 10))

        self.run_tab = ScrollableFrame(self.notebook)
        self.test_tab = ScrollableFrame(self.notebook)
        self.analysis_tab = ScrollableFrame(self.notebook)
        self.notebook.add(self.run_tab, text="运行")
        self.notebook.add(self.test_tab, text="测试")
        self.notebook.add(self.analysis_tab, text="工作台")

        self._build_run_tab(self.run_tab.content)
        self._build_test_tab(self.test_tab.content)
        self._build_analysis_tab(self.analysis_tab.content)

    def _build_run_tab(self, parent: ttk.Frame) -> None:
        # Vars
        self.work_dir_var = tk.StringVar()
        self.video_var = tk.StringVar()
        self.video_sample_fps_var = tk.StringVar(value="2.0")
        self.video_max_frames_var = tk.StringVar()
        self.video_blur_threshold_var = tk.StringVar()
        self.video_dedupe_threshold_var = tk.StringVar()
        self.video_min_gap_sec_var = tk.StringVar()
        self.colmap_var = tk.StringVar(value=_detect_colmap_bin())
        self.pipeline_var = tk.StringVar(value="superpoint+lightglue")
        self.dense_dir_var = tk.StringVar()
        self.gpu_var = tk.StringVar()
        self.pm_gpu_var = tk.StringVar()
        self.dim_quality_var = tk.StringVar(value="medium")
        self.dim_camera_model_var = tk.StringVar(value="simple-radial")
        self.mode_var = tk.StringVar(value=MODE_FULL)

        self.use_dim_env_var = tk.BooleanVar(value=True)
        self.skip_dim_var = tk.BooleanVar(value=False)
        self.overwrite_var = tk.BooleanVar(value=False)
        self.dim_multi_camera_var = tk.BooleanVar(value=False)
        self.skip_geom_verify_var = tk.BooleanVar(value=False)

        project = ttk.LabelFrame(parent, text="项目", style="Section.TLabelframe", padding=10)
        project.pack(fill=tk.X, pady=(0, 10))
        self._grid_labeled_entry(project, "工作目录 (输出目录):", self.work_dir_var, row=0, browse="dir")
        self._grid_labeled_entry(project, "COLMAP 可执行文件:", self.colmap_var, row=1, browse="file")
        self._grid_labeled_entry(project, "Dense 输出目录(可空):", self.dense_dir_var, row=2, browse="dir")

        video = ttk.LabelFrame(parent, text="视频输入（可空）", style="Section.TLabelframe", padding=10)
        video.pack(fill=tk.X, pady=(0, 10))
        self.video_entry = self._grid_labeled_entry(video, "视频文件:", self.video_var, row=0, browse="file")
        self.video_fps_entry = self._grid_labeled_entry(
            video,
            "抽帧 FPS (默认 2.0):",
            self.video_sample_fps_var,
            row=1,
        )
        self.video_max_frames_entry = self._grid_labeled_entry(
            video,
            "最大帧数(可空):",
            self.video_max_frames_var,
            row=2,
        )
        self.video_blur_entry = self._grid_labeled_entry(
            video,
            "最小清晰度(可空):",
            self.video_blur_threshold_var,
            row=3,
        )
        self.video_dedupe_entry = self._grid_labeled_entry(
            video,
            "去重阈值(可空):",
            self.video_dedupe_threshold_var,
            row=4,
        )
        self.video_min_gap_entry = self._grid_labeled_entry(
            video,
            "最小时间间隔秒(可空):",
            self.video_min_gap_sec_var,
            row=5,
        )

        options = ttk.LabelFrame(parent, text="运行模式与匹配", style="Section.TLabelframe", padding=10)
        options.pack(fill=tk.X, pady=(0, 10))

        self.mode_combo = self._grid_labeled_combo(
            options,
            "模式:",
            self.mode_var,
            values=(MODE_FULL, MODE_DIM_ONLY, MODE_DENSE_ONLY),
            row=0,
        )
        self.pipeline_combo = self._grid_labeled_combo(
            options,
            "DIM pipeline:",
            self.pipeline_var,
            values=PIPELINE_PRESETS,
            row=1,
            editable=True,
        )
        self.dim_quality_combo = self._grid_labeled_combo(
            options, "DIM quality:", self.dim_quality_var, values=DIM_QUALITY_OPTIONS, row=2
        )
        self.dim_camera_model_combo = self._grid_labeled_combo(
            options, "DIM camera model:", self.dim_camera_model_var, values=CAMERA_MODEL_OPTIONS, row=3
        )

        gpu = ttk.LabelFrame(parent, text="GPU（可空）", style="Section.TLabelframe", padding=10)
        gpu.pack(fill=tk.X, pady=(0, 10))
        self.dim_gpu_entry = self._grid_labeled_entry(gpu, "DIM GPU index:", self.gpu_var, row=0)
        self.pm_gpu_entry = self._grid_labeled_entry(gpu, "PatchMatch GPU index:", self.pm_gpu_var, row=1)

        flags = ttk.LabelFrame(parent, text="开关", style="Section.TLabelframe", padding=10)
        flags.pack(fill=tk.X, pady=(0, 10))
        self.use_dim_env_check = ttk.Checkbutton(
            flags, text="Use managed Py3.9 DIM env (conda)", variable=self.use_dim_env_var
        )
        self.use_dim_env_check.grid(
            row=0, column=0, sticky="w", padx=(0, 18), pady=2
        )
        self.overwrite_check = ttk.Checkbutton(flags, text="覆盖输出 (--overwrite)", variable=self.overwrite_var)
        self.overwrite_check.grid(
            row=0, column=1, sticky="w", padx=(0, 18), pady=2
        )
        self.skip_dim_check = ttk.Checkbutton(flags, text="跳过 DIM (--skip_dim)", variable=self.skip_dim_var)
        self.skip_dim_check.grid(
            row=1, column=0, sticky="w", padx=(0, 18), pady=2
        )
        self.multi_cam_check = ttk.Checkbutton(flags, text="DIM 多相机", variable=self.dim_multi_camera_var)
        self.multi_cam_check.grid(
            row=1, column=1, sticky="w", padx=(0, 18), pady=2
        )
        self.skip_geom_check = ttk.Checkbutton(flags, text="跳过 geom verify", variable=self.skip_geom_verify_var)
        self.skip_geom_check.grid(
            row=2, column=0, sticky="w", padx=(0, 18), pady=2
        )

        actions = ttk.Frame(parent)
        actions.pack(fill=tk.X, pady=(0, 6))
        self.run_btn = ttk.Button(actions, text="开始运行", command=lambda: self.run_pipeline_thread(analyze_after=False))
        self.run_btn.pack(side=tk.LEFT)
        self.run_and_analyze_btn = ttk.Button(
            actions,
            text="生成点云并分析",
            command=lambda: self.run_pipeline_thread(analyze_after=True),
        )
        self.run_and_analyze_btn.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(
            actions,
            text="提示：填了视频文件时会先抽帧到 work_dir/images；可选的清晰度/去重参数会先筛关键帧；“生成点云并分析”会继续生成四视图并调用 OpenAI。",
        ).pack(side=tk.LEFT, padx=12)

        self.mode_var.trace_add("write", lambda *_: self._sync_mode_ui())
        self._sync_mode_ui()

    def _build_test_tab(self, parent: ttk.Frame) -> None:
        self.test_pipelines_var = tk.StringVar(value="all")
        self.test_max_images_var = tk.StringVar(value="")
        self.test_quality_var = tk.StringVar(value="low")
        self.benchmark_var = tk.BooleanVar(value=True)
        self.benchmark_interval_var = tk.StringVar(value="0.2")
        self.test_run_dense_var = tk.BooleanVar(value=True)

        top = ttk.LabelFrame(parent, text="DIM pipelines 测试", style="Section.TLabelframe", padding=10)
        top.pack(fill=tk.X, pady=(0, 10))

        self._grid_labeled_entry(top, "pipelines (all/逗号分隔):", self.test_pipelines_var, row=0)
        self._grid_labeled_entry(top, "限制图片数(可空，使用全部):", self.test_max_images_var, row=1)
        self._grid_labeled_combo(top, "DIM quality:", self.test_quality_var, values=DIM_QUALITY_OPTIONS, row=2)
        ttk.Checkbutton(top, text="生成对比报告（benchmark）", variable=self.benchmark_var).grid(
            row=3, column=0, sticky="w", pady=(6, 2)
        )
        ttk.Checkbutton(top, text="测试后生成点云（Dense）", variable=self.test_run_dense_var).grid(
            row=4, column=0, sticky="w", pady=(2, 2)
        )
        self._grid_labeled_entry(top, "benchmark RSS 采样间隔(s):", self.benchmark_interval_var, row=5)

        btns = ttk.Frame(top)
        btns.grid(row=7, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.preset_pipelines_btn = ttk.Button(btns, text="填入推荐列表", command=self.fill_recommended_pipelines)
        self.preset_pipelines_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.list_pipelines_btn = ttk.Button(btns, text="列出 DIM pipelines", command=self.list_pipelines_thread)
        self.list_pipelines_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.probe_pipelines_btn = ttk.Button(btns, text="Probe pipelines", command=self.probe_pipelines_thread)
        self.probe_pipelines_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.test_pipelines_btn = ttk.Button(btns, text="跑测试", command=self.test_pipelines_thread)
        self.test_pipelines_btn.pack(side=tk.LEFT)

        hint = ttk.Label(
            parent,
            text=(
                "建议：勾选“生成对比报告（benchmark）”，跑完后会在输出目录生成 benchmark.csv / benchmark.json。"
                "数据量不大时可直接全量跑；需要加速时再用“限制图片数”。"
                "勾选“测试后生成点云（Dense）”会为每个 pipeline 输出 dense/fused.ply。"
            ),
            wraplength=860,
        )
        hint.pack(fill=tk.X)

    def _build_analysis_tab(self, parent: ttk.Frame) -> None:
        self.analysis_point_cloud_var = tk.StringVar()
        self.analysis_snapshot_var = tk.StringVar()
        self.analysis_response_var = tk.StringVar()
        self.openai_api_key_var = tk.StringVar()
        self.openai_base_url_var = tk.StringVar(value=os.environ.get(OPENAI_BASE_URL_ENV, DEFAULT_OPENAI_BASE_URL))
        self.openai_model_var = tk.StringVar(value=DEFAULT_OPENAI_MODEL)
        self.openai_detail_var = tk.StringVar(value="low")
        self.preview_yaw_var = tk.DoubleVar(value=35.0)
        self.preview_pitch_var = tk.DoubleVar(value=-25.0)
        self.preview_status_var = tk.StringVar(value="等待点云。")
        self.api_view_status_var = tk.StringVar(value="尚未生成四张 API 图片。")

        header = ttk.Label(
            parent,
            text="从左到右：视频/点云输入 -> 可旋转点云预览 -> 四张 API 视图 -> API 输出。",
        )
        header.pack(fill=tk.X, pady=(0, 8))

        workspace = ttk.Frame(parent)
        workspace.pack(fill=tk.BOTH, expand=True)
        workspace.grid_columnconfigure(0, weight=2)
        workspace.grid_columnconfigure(1, weight=3)
        workspace.grid_columnconfigure(2, weight=3)
        workspace.grid_columnconfigure(3, weight=3)
        workspace.grid_rowconfigure(0, weight=1)

        left = ttk.LabelFrame(workspace, text="1. 输入与运行", style="Section.TLabelframe", padding=10)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.workbench_work_dir_entry = self._grid_labeled_entry(left, "工作目录:", self.work_dir_var, row=0, browse="dir")
        self.workbench_video_entry = self._grid_labeled_entry(left, "视频文件:", self.video_var, row=1, browse="file")
        self.analysis_point_cloud_entry = self._grid_labeled_entry(
            left,
            "点云文件 (.ply):",
            self.analysis_point_cloud_var,
            row=2,
            browse="file",
        )
        self.workbench_colmap_entry = self._grid_labeled_entry(left, "COLMAP:", self.colmap_var, row=3, browse="file")
        self.workbench_pipeline_combo = self._grid_labeled_combo(
            left,
            "DIM pipeline:",
            self.pipeline_var,
            values=PIPELINE_PRESETS,
            row=4,
            editable=True,
        )
        self.workbench_quality_combo = self._grid_labeled_combo(
            left,
            "DIM quality:",
            self.dim_quality_var,
            values=DIM_QUALITY_OPTIONS,
            row=5,
        )
        self.workbench_video_fps_entry = self._grid_labeled_entry(
            left,
            "抽帧 FPS:",
            self.video_sample_fps_var,
            row=6,
        )
        self.workbench_video_max_frames_entry = self._grid_labeled_entry(
            left,
            "最大帧数(可空):",
            self.video_max_frames_var,
            row=7,
        )

        workbench_flags = ttk.Frame(left)
        workbench_flags.grid(row=8, column=0, columnspan=3, sticky="w", pady=(6, 6))
        self.workbench_overwrite_check = ttk.Checkbutton(
            workbench_flags,
            text="覆盖输出 (--overwrite)",
            variable=self.overwrite_var,
        )
        self.workbench_overwrite_check.pack(side=tk.LEFT)
        self.workbench_dim_env_check = ttk.Checkbutton(
            workbench_flags,
            text="Use managed DIM env",
            variable=self.use_dim_env_var,
        )
        self.workbench_dim_env_check.pack(side=tk.LEFT, padx=(12, 0))

        run_actions = ttk.Frame(left)
        run_actions.grid(row=9, column=0, columnspan=3, sticky="we", pady=(4, 6))
        run_actions.grid_columnconfigure(0, weight=1)
        run_actions.grid_columnconfigure(1, weight=1)
        self.workbench_run_btn = ttk.Button(run_actions, text="生成点云", command=self.run_workbench_thread)
        self.workbench_run_btn.grid(row=0, column=0, sticky="we", padx=(0, 4))
        self.workbench_run_analyze_btn = ttk.Button(
            run_actions,
            text="生成点云并分析",
            command=lambda: self.run_workbench_thread(analyze_after=True),
        )
        self.workbench_run_analyze_btn.grid(row=0, column=1, sticky="we", padx=(4, 0))

        asset_actions = ttk.Frame(left)
        asset_actions.grid(row=10, column=0, columnspan=3, sticky="we", pady=(0, 6))
        asset_actions.grid_columnconfigure(0, weight=1)
        asset_actions.grid_columnconfigure(1, weight=1)
        self.snapshot_btn = ttk.Button(asset_actions, text="生成四图", command=self.snapshot_point_cloud_thread)
        self.snapshot_btn.grid(row=0, column=0, sticky="we", padx=(0, 4))
        self.analyze_btn = ttk.Button(asset_actions, text="四图发给 API", command=self.analyze_point_cloud_thread)
        self.analyze_btn.grid(row=0, column=1, sticky="we", padx=(4, 0))

        self.workbench_hint_label = ttk.Label(
            left,
            text=(
                "这个工作台固定走全流程生成点云；如果你已经有 `.ply`，直接在上面选择点云文件，"
                "再点“生成四图”或“四图发给 API”。"
            ),
            wraplength=260,
            justify="left",
        )
        self.workbench_hint_label.grid(row=11, column=0, columnspan=3, sticky="we", pady=(4, 0))

        preview = ttk.LabelFrame(workspace, text="2. 点云预览（可转动）", style="Section.TLabelframe", padding=10)
        preview.grid(row=0, column=1, sticky="nsew", padx=8)
        preview.grid_columnconfigure(0, weight=1)
        preview.grid_rowconfigure(0, weight=1)
        self.point_cloud_preview_label = ttk.Label(preview, text="尚未生成点云预览。", anchor="center")
        self.point_cloud_preview_label.grid(row=0, column=0, sticky="nsew")

        preview_controls = ttk.Frame(preview)
        preview_controls.grid(row=1, column=0, sticky="we", pady=(10, 0))
        preview_controls.grid_columnconfigure(1, weight=1)

        ttk.Label(preview_controls, text="Yaw").grid(row=0, column=0, sticky="w")
        self.preview_yaw_scale = ttk.Scale(preview_controls, from_=-180, to=180, variable=self.preview_yaw_var)
        self.preview_yaw_scale.grid(row=0, column=1, sticky="we", padx=(8, 8))
        self.preview_yaw_value = ttk.Label(preview_controls, text="35°")
        self.preview_yaw_value.grid(row=0, column=2, sticky="e")

        ttk.Label(preview_controls, text="Pitch").grid(row=1, column=0, sticky="w")
        self.preview_pitch_scale = ttk.Scale(preview_controls, from_=-90, to=90, variable=self.preview_pitch_var)
        self.preview_pitch_scale.grid(row=1, column=1, sticky="we", padx=(8, 8))
        self.preview_pitch_value = ttk.Label(preview_controls, text="-25°")
        self.preview_pitch_value.grid(row=1, column=2, sticky="e")

        preview_buttons = ttk.Frame(preview)
        preview_buttons.grid(row=2, column=0, sticky="we", pady=(8, 0))
        preview_buttons.grid_columnconfigure(0, weight=1)
        preview_buttons.grid_columnconfigure(1, weight=1)
        preview_buttons.grid_columnconfigure(2, weight=1)
        preview_buttons.grid_columnconfigure(3, weight=1)
        preview_buttons.grid_columnconfigure(4, weight=1)
        self.preview_left_btn = ttk.Button(preview_buttons, text="左转", command=lambda: self._nudge_preview(yaw_delta=-15.0))
        self.preview_left_btn.grid(row=0, column=0, sticky="we", padx=(0, 4))
        self.preview_right_btn = ttk.Button(preview_buttons, text="右转", command=lambda: self._nudge_preview(yaw_delta=15.0))
        self.preview_right_btn.grid(row=0, column=1, sticky="we", padx=4)
        self.preview_up_btn = ttk.Button(preview_buttons, text="抬高", command=lambda: self._nudge_preview(pitch_delta=10.0))
        self.preview_up_btn.grid(row=0, column=2, sticky="we", padx=4)
        self.preview_down_btn = ttk.Button(preview_buttons, text="压低", command=lambda: self._nudge_preview(pitch_delta=-10.0))
        self.preview_down_btn.grid(row=0, column=3, sticky="we", padx=4)
        self.preview_refresh_btn = ttk.Button(preview_buttons, text="刷新", command=self.refresh_point_cloud_preview_thread)
        self.preview_refresh_btn.grid(row=0, column=4, sticky="we", padx=(4, 0))
        self.preview_reset_btn = ttk.Button(preview_buttons, text="重置视角", command=self.reset_point_cloud_preview_thread)
        self.preview_reset_btn.grid(row=1, column=0, columnspan=5, sticky="we", pady=(6, 0))

        self.preview_status_label = ttk.Label(
            preview,
            textvariable=self.preview_status_var,
            wraplength=420,
            justify="left",
        )
        self.preview_status_label.grid(row=3, column=0, sticky="we", pady=(8, 0))

        api_views = ttk.LabelFrame(workspace, text="3. 发给 API 的四张视图", style="Section.TLabelframe", padding=10)
        api_views.grid(row=0, column=2, sticky="nsew", padx=8)
        api_views.grid_columnconfigure(0, weight=1)
        api_views.grid_columnconfigure(1, weight=1)
        api_views.grid_rowconfigure(0, weight=1)
        api_views.grid_rowconfigure(1, weight=1)
        self.api_view_labels: dict[str, ttk.Label] = {}
        for index, (view_name, _, _) in enumerate(DEFAULT_POINT_CLOUD_API_VIEWS):
            holder = ttk.LabelFrame(api_views, text=view_name, padding=6)
            holder.grid(row=index // 2, column=index % 2, sticky="nsew", padx=4, pady=4)
            holder.grid_columnconfigure(0, weight=1)
            holder.grid_rowconfigure(0, weight=1)
            label = ttk.Label(holder, text=f"{view_name}\n尚未生成", anchor="center")
            label.grid(row=0, column=0, sticky="nsew")
            self.api_view_labels[view_name] = label
        self.api_view_status_label = ttk.Label(
            api_views,
            textvariable=self.api_view_status_var,
            wraplength=420,
            justify="left",
        )
        self.api_view_status_label.grid(row=2, column=0, columnspan=2, sticky="we", padx=4, pady=(8, 0))

        output = ttk.LabelFrame(workspace, text="4. API 输出", style="Section.TLabelframe", padding=10)
        output.grid(row=0, column=3, sticky="nsew", padx=(8, 0))
        output.grid_columnconfigure(1, weight=1)
        output.grid_rowconfigure(5, weight=1)
        output.grid_rowconfigure(7, weight=2)
        self.openai_api_key_entry = self._grid_labeled_entry(output, "API Key:", self.openai_api_key_var, row=0)
        self.openai_api_key_entry.configure(show="*")
        self.openai_base_url_entry = self._grid_labeled_entry(output, "API Base URL:", self.openai_base_url_var, row=1)
        self.openai_model_entry = self._grid_labeled_entry(output, "Model:", self.openai_model_var, row=2)
        self.openai_detail_combo = self._grid_labeled_combo(
            output,
            "图像细节:",
            self.openai_detail_var,
            values=OPENAI_IMAGE_DETAIL_OPTIONS,
            row=3,
        )
        self.analysis_response_entry = self._grid_labeled_entry(output, "API 响应 JSON:", self.analysis_response_var, row=4)
        ttk.Label(output, text="分析提示词:").grid(row=5, column=0, sticky="nw", pady=4)
        self.analysis_prompt_box = ScrolledText(output, height=10, wrap="word")
        self.analysis_prompt_box.grid(row=5, column=1, columnspan=2, sticky="nsew", pady=4, padx=(8, 0))

        output_actions = ttk.Frame(output)
        output_actions.grid(row=6, column=0, columnspan=3, sticky="we", pady=(8, 4))
        ttk.Label(
            output_actions,
            text="留空 API Key 时会回退到环境变量 OPENAI_API_KEY；API key 不会写入本地配置。",
            wraplength=360,
            justify="left",
        ).pack(anchor="w")

        ttk.Label(output, text="API 回答:").grid(row=7, column=0, sticky="nw", pady=(8, 0))
        self.analysis_result_box = ScrolledText(output, height=16, wrap="word", state="disabled")
        self.analysis_result_box.grid(row=7, column=1, columnspan=2, sticky="nsew", pady=(8, 0), padx=(8, 0))

        self.preview_yaw_scale.bind("<ButtonRelease-1>", lambda _event: self.refresh_point_cloud_preview_thread())
        self.preview_pitch_scale.bind("<ButtonRelease-1>", lambda _event: self.refresh_point_cloud_preview_thread())

    def _grid_labeled_entry(
        self,
        parent: ttk.Frame,
        label: str,
        var: tk.StringVar,
        row: int,
        *,
        browse: str | None = None,
    ) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=1, sticky="we", pady=4, padx=(8, 8))
        parent.grid_columnconfigure(1, weight=1)
        if browse:
            if browse == "dir":
                cmd = lambda: self._choose_dir(var)
                text = "浏览..."
            else:
                cmd = lambda: self._choose_file(var)
                text = "选择..."
            ttk.Button(parent, text=text, command=cmd, width=10).grid(row=row, column=2, pady=4)
        return entry

    def _grid_labeled_combo(
        self,
        parent: ttk.Frame,
        label: str,
        var: tk.StringVar,
        *,
        values: tuple[str, ...],
        row: int,
        editable: bool = False,
    ) -> ttk.Combobox:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        state = "normal" if editable else "readonly"
        combo = ttk.Combobox(parent, textvariable=var, values=values, state=state)
        combo.grid(row=row, column=1, sticky="we", pady=4, padx=(8, 8))
        parent.grid_columnconfigure(1, weight=1)
        ttk.Button(parent, text="?", width=3, command=lambda: self._show_help_for(label)).grid(
            row=row, column=2, pady=4
        )
        return combo

    def _build_log(self) -> None:
        frame = ttk.LabelFrame(self.outer, text="日志", style="Section.TLabelframe", padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="日志输出：").pack(anchor="w")
        log_box = ScrolledText(frame, height=18, wrap="word", state="disabled")
        log_box.pack(fill=tk.BOTH, expand=True, pady=(6, 0))
        self.log_box = log_box

    def clear_log(self) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", tk.END)
        self.log_box.configure(state="disabled")

    def _choose_dir(self, var: tk.StringVar) -> None:
        path = filedialog.askdirectory(title="选择目录")
        if path:
            var.set(path)

    def _choose_file(self, var: tk.StringVar) -> None:
        path = filedialog.askopenfilename(title="选择文件")
        if path:
            var.set(path)

    def _set_scrolled_text(self, widget: ScrolledText, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _set_prompt_text(self, text: str) -> None:
        self.analysis_prompt_box.delete("1.0", tk.END)
        self.analysis_prompt_box.insert("1.0", text)

    def _get_prompt_text(self) -> str:
        return self.analysis_prompt_box.get("1.0", tk.END).strip()

    def _set_analysis_output(self, text: str) -> None:
        self._set_scrolled_text(self.analysis_result_box, text)

    def _update_preview_angle_labels(self) -> None:
        self.preview_yaw_value.configure(text=f"{self.preview_yaw_var.get():.0f}°")
        self.preview_pitch_value.configure(text=f"{self.preview_pitch_var.get():.0f}°")

    def _set_point_cloud_preview(self, image_path: str | None) -> None:
        self._update_preview_angle_labels()
        if not image_path:
            self.preview_photo = None
            self.point_cloud_preview_label.configure(text="尚未生成点云预览。", image="")
            return
        try:
            photo = tk.PhotoImage(file=image_path)
        except tk.TclError:
            self.preview_photo = None
            self.point_cloud_preview_label.configure(text=f"预览已生成，但 Tk 无法显示：{image_path}", image="")
            return
        self.preview_photo = photo
        self.point_cloud_preview_label.configure(text="", image=self.preview_photo)

    def _clear_api_view_previews(self) -> None:
        self.analysis_api_view_paths = []
        self.api_view_photos = {}
        for view_name, label in self.api_view_labels.items():
            label.configure(text=f"{view_name}\n尚未生成", image="")
        self.api_view_status_var.set("尚未生成四张 API 图片。")

    def _set_api_view_previews(self, view_names: list[str], image_paths: list[str]) -> None:
        self.analysis_api_view_paths = list(image_paths)
        self.api_view_photos = {}
        shown = 0
        for view_name, image_path in zip(view_names, image_paths, strict=False):
            label = self.api_view_labels.get(view_name)
            if label is None:
                continue
            try:
                photo = tk.PhotoImage(file=image_path)
            except tk.TclError:
                label.configure(text=f"{view_name}\n已生成：{image_path}", image="")
                continue
            self.api_view_photos[view_name] = photo
            label.configure(text="", image=photo)
            shown += 1
        self.api_view_status_var.set(
            f"已生成 {len(image_paths)} 张视图，当前显示 {shown} 张。发送 API 时会按 Front / Right / Top / Isometric 顺序上传。"
        )

    def _analysis_artifacts_dir(self, point_cloud_path: str) -> Path:
        point_cloud = Path(point_cloud_path).expanduser().resolve()
        if point_cloud.parent.name == "dense" and point_cloud.parent.parent.exists():
            return point_cloud.parent.parent / "analysis"
        return point_cloud.parent / "analysis"

    def _set_point_cloud_result(self, point_cloud_path: str) -> None:
        self.analysis_point_cloud_var.set(point_cloud_path)
        self.analysis_snapshot_var.set("")
        self.analysis_response_var.set("")
        self._render_data_cache = None
        self._render_data_cache_path = None
        self._set_point_cloud_preview(None)
        self._clear_api_view_previews()
        self.preview_status_var.set("点云已更新。点击“刷新”可查看当前角度。")
        self._set_analysis_output("")

    def _get_render_data(self, point_cloud_path: str) -> PointCloudRenderData:
        resolved_path = str(Path(point_cloud_path).expanduser().resolve())
        if self._render_data_cache is not None and self._render_data_cache_path == resolved_path:
            return self._render_data_cache
        render_data = prepare_point_cloud_render_data(point_cloud_path=resolved_path)
        self._render_data_cache = render_data
        self._render_data_cache_path = resolved_path
        return render_data

    def _nudge_preview(self, *, yaw_delta: float = 0.0, pitch_delta: float = 0.0) -> None:
        self.preview_yaw_var.set(self.preview_yaw_var.get() + yaw_delta)
        self.preview_pitch_var.set(max(-90.0, min(90.0, self.preview_pitch_var.get() + pitch_delta)))
        self.refresh_point_cloud_preview_thread()

    def _collect_openai_settings(self) -> tuple[str, str, str, str, str]:
        prompt = self._get_prompt_text()
        if not prompt:
            raise ValueError("分析提示词不能为空。")
        api_key = self.openai_api_key_var.get().strip()
        base_url = self.openai_base_url_var.get().strip() or os.environ.get(OPENAI_BASE_URL_ENV, DEFAULT_OPENAI_BASE_URL)
        model = self.openai_model_var.get().strip() or DEFAULT_OPENAI_MODEL
        detail = self.openai_detail_var.get().strip() or "low"
        return prompt, api_key, base_url, model, detail

    def _collect_analysis_request(self) -> tuple[str, str, str, str, str, str]:
        point_cloud_path = self.analysis_point_cloud_var.get().strip()
        if not point_cloud_path:
            raise ValueError("请先提供点云文件，或先运行生成点云。")
        prompt, api_key, base_url, model, detail = self._collect_openai_settings()
        return point_cloud_path, prompt, api_key, base_url, model, detail

    def refresh_point_cloud_preview_thread(self) -> None:
        if self.running:
            return
        point_cloud_path = self.analysis_point_cloud_var.get().strip()
        if not point_cloud_path:
            messagebox.showerror("参数错误", "请先提供点云文件，或先运行生成点云。")
            return

        self.running = True
        self._set_busy(True)
        self._log("===== 刷新点云预览 =====")

        def worker() -> None:
            try:
                preview_path = self._render_current_preview(point_cloud_path)
                self.root.after(0, lambda: self._set_point_cloud_preview(preview_path))
            except Exception as e:  # noqa: BLE001
                self._log(f"[ERROR] {e}")
            finally:
                self.root.after(0, self._on_finished)

        threading.Thread(target=worker, daemon=True).start()

    def reset_point_cloud_preview_thread(self) -> None:
        self.preview_yaw_var.set(35.0)
        self.preview_pitch_var.set(-25.0)
        self.refresh_point_cloud_preview_thread()

    def run_workbench_thread(self, analyze_after: bool = False) -> None:
        self.mode_var.set(MODE_FULL)
        self.run_pipeline_thread(analyze_after=analyze_after)

    def _str_to_int(self, value: str) -> int | None:
        value = value.strip()
        return int(value) if value else None

    def _str_to_float(self, value: str) -> float | None:
        value = value.strip()
        return float(value) if value else None

    def _get_video_params(self) -> tuple[str | None, float, int | None, float, float, float]:
        video_path = self.video_var.get().strip() or None
        sample_fps = self._str_to_float(self.video_sample_fps_var.get() or "2.0")
        if sample_fps is None or sample_fps <= 0:
            raise ValueError("抽帧 FPS 必须是大于 0 的数字。")
        max_frames = self._str_to_int(self.video_max_frames_var.get())
        if max_frames is not None and max_frames <= 0:
            raise ValueError("最大帧数必须是正整数或留空。")
        blur_threshold = self._str_to_float(self.video_blur_threshold_var.get()) or 0.0
        dedupe_threshold = self._str_to_float(self.video_dedupe_threshold_var.get()) or 0.0
        min_gap_sec = self._str_to_float(self.video_min_gap_sec_var.get()) or 0.0
        if blur_threshold < 0:
            raise ValueError("最小清晰度不能小于 0。")
        if dedupe_threshold < 0:
            raise ValueError("去重阈值不能小于 0。")
        if min_gap_sec < 0:
            raise ValueError("最小时间间隔秒不能小于 0。")
        return video_path, sample_fps, max_frames, blur_threshold, dedupe_threshold, min_gap_sec

    def _set_busy(self, busy: bool) -> None:
        if busy:
            for w in (
                self.run_btn,
                self.run_and_analyze_btn,
                self.workbench_run_btn,
                self.workbench_run_analyze_btn,
                self.preset_pipelines_btn,
                self.list_pipelines_btn,
                self.probe_pipelines_btn,
                self.test_pipelines_btn,
                self.snapshot_btn,
                self.analyze_btn,
                self.preview_left_btn,
                self.preview_right_btn,
                self.preview_up_btn,
                self.preview_down_btn,
                self.preview_refresh_btn,
                self.preview_reset_btn,
                self.mode_combo,
                self.pipeline_combo,
                self.workbench_pipeline_combo,
                self.dim_quality_combo,
                self.workbench_quality_combo,
                self.dim_camera_model_combo,
                self.dim_gpu_entry,
                self.pm_gpu_entry,
                self.video_entry,
                self.workbench_video_entry,
                self.video_fps_entry,
                self.workbench_video_fps_entry,
                self.video_max_frames_entry,
                self.workbench_video_max_frames_entry,
                self.video_blur_entry,
                self.video_dedupe_entry,
                self.video_min_gap_entry,
                self.workbench_work_dir_entry,
                self.workbench_colmap_entry,
                self.analysis_point_cloud_entry,
                self.analysis_response_entry,
                self.openai_api_key_entry,
                self.openai_base_url_entry,
                self.openai_model_entry,
                self.openai_detail_combo,
                self.use_dim_env_check,
                self.workbench_dim_env_check,
                self.overwrite_check,
                self.workbench_overwrite_check,
                self.skip_dim_check,
                self.multi_cam_check,
                self.skip_geom_check,
            ):
                w.configure(state="disabled")
            return

        # Restore interactive state.
        self.run_btn.configure(state="normal")
        self.run_and_analyze_btn.configure(state="normal")
        self.workbench_run_btn.configure(state="normal")
        self.workbench_run_analyze_btn.configure(state="normal")
        self.preset_pipelines_btn.configure(state="normal")
        self.list_pipelines_btn.configure(state="normal")
        self.probe_pipelines_btn.configure(state="normal")
        self.test_pipelines_btn.configure(state="normal")
        self.snapshot_btn.configure(state="normal")
        self.analyze_btn.configure(state="normal")
        self.preview_left_btn.configure(state="normal")
        self.preview_right_btn.configure(state="normal")
        self.preview_up_btn.configure(state="normal")
        self.preview_down_btn.configure(state="normal")
        self.preview_refresh_btn.configure(state="normal")
        self.preview_reset_btn.configure(state="normal")

        self.mode_combo.configure(state="readonly")
        self.pipeline_combo.configure(state="normal")
        self.workbench_pipeline_combo.configure(state="normal")
        self.dim_quality_combo.configure(state="readonly")
        self.workbench_quality_combo.configure(state="readonly")
        self.dim_camera_model_combo.configure(state="readonly")
        self.dim_gpu_entry.configure(state="normal")
        self.pm_gpu_entry.configure(state="normal")
        self.video_entry.configure(state="normal")
        self.workbench_video_entry.configure(state="normal")
        self.video_fps_entry.configure(state="normal")
        self.workbench_video_fps_entry.configure(state="normal")
        self.video_max_frames_entry.configure(state="normal")
        self.workbench_video_max_frames_entry.configure(state="normal")
        self.video_blur_entry.configure(state="normal")
        self.video_dedupe_entry.configure(state="normal")
        self.video_min_gap_entry.configure(state="normal")
        self.workbench_work_dir_entry.configure(state="normal")
        self.workbench_colmap_entry.configure(state="normal")
        self.analysis_point_cloud_entry.configure(state="normal")
        self.analysis_response_entry.configure(state="normal")
        self.openai_api_key_entry.configure(state="normal")
        self.openai_base_url_entry.configure(state="normal")
        self.openai_model_entry.configure(state="normal")
        self.openai_detail_combo.configure(state="readonly")
        self.use_dim_env_check.configure(state="normal")
        self.workbench_dim_env_check.configure(state="normal")
        self.overwrite_check.configure(state="normal")
        self.workbench_overwrite_check.configure(state="normal")
        self.skip_dim_check.configure(state="normal")
        self.multi_cam_check.configure(state="normal")
        self.skip_geom_check.configure(state="normal")
        self._sync_mode_ui()

    def _sync_mode_ui(self) -> None:
        mode = self.mode_var.get().strip()
        if mode == MODE_DENSE_ONLY:
            self.skip_dim_var.set(True)
            self.skip_dim_check.configure(state="disabled")
            self.pipeline_combo.configure(state="disabled")
            self.dim_quality_combo.configure(state="disabled")
            self.dim_camera_model_combo.configure(state="disabled")
            self.multi_cam_check.configure(state="disabled")
            self.skip_geom_check.configure(state="disabled")
            self.pm_gpu_entry.configure(state="normal")
            self.video_entry.configure(state="disabled")
            self.video_fps_entry.configure(state="disabled")
            self.video_max_frames_entry.configure(state="disabled")
            self.video_blur_entry.configure(state="disabled")
            self.video_dedupe_entry.configure(state="disabled")
            self.video_min_gap_entry.configure(state="disabled")
        elif mode == MODE_DIM_ONLY:
            self.skip_dim_var.set(False)
            self.skip_dim_check.configure(state="disabled")
            self.pipeline_combo.configure(state="normal")
            self.dim_quality_combo.configure(state="readonly")
            self.dim_camera_model_combo.configure(state="readonly")
            self.multi_cam_check.configure(state="normal")
            self.skip_geom_check.configure(state="normal")
            self.pm_gpu_entry.configure(state="disabled")
            self.video_entry.configure(state="normal")
            self.video_fps_entry.configure(state="normal")
            self.video_max_frames_entry.configure(state="normal")
            self.video_blur_entry.configure(state="normal")
            self.video_dedupe_entry.configure(state="normal")
            self.video_min_gap_entry.configure(state="normal")
        else:
            self.skip_dim_check.configure(state="normal")
            self.pipeline_combo.configure(state="normal")
            self.dim_quality_combo.configure(state="readonly")
            self.dim_camera_model_combo.configure(state="readonly")
            self.multi_cam_check.configure(state="normal")
            self.skip_geom_check.configure(state="normal")
            self.pm_gpu_entry.configure(state="normal")
            self.video_entry.configure(state="normal")
            self.video_fps_entry.configure(state="normal")
            self.video_max_frames_entry.configure(state="normal")
            self.video_blur_entry.configure(state="normal")
            self.video_dedupe_entry.configure(state="normal")
            self.video_min_gap_entry.configure(state="normal")

    def _show_help_for(self, label: str) -> None:
        txt = {
            "模式:": (
                f"{MODE_FULL}: 先跑 DIM 导出 + COLMAP mapper 得到 sparse，再跑 dense。\n"
                f"{MODE_DIM_ONLY}: 只生成 sparse（适合先看能否稀疏重建）。\n"
                f"{MODE_DENSE_ONLY}: 只跑 dense（需要 work_dir 下已有 sparse/；视频输入在该模式下不可用）。"
            ),
            "DIM pipeline:": "选择匹配模型组合；常用推荐：aliked+lightglue / sift+lightglue / loftr。",
            "DIM quality:": "分辨率预设；high/highest 更稳但更慢，lowest 用于快速测试。",
            "DIM camera model:": "写入 COLMAP 数据库的相机模型；单相机 UAV 通常 simple-radial 即可。",
            "API Base URL:": "默认是官方 OpenAI `/v1`；如果你用兼容网关，也可以填自己的 `.../v1` 地址。",
            "图像细节:": "OpenAI 图像输入 detail；low 更快更省，high 更细，auto 让 API 自己判断。",
        }.get(label, "暂无帮助信息。")
        messagebox.showinfo("帮助", txt)

    def _log(self, msg: str) -> None:
        self.log_queue.put(msg)

    def _poll_logs(self) -> None:
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self.log_box.configure(state="normal")
                self.log_box.insert(tk.END, msg + "\n")
                self.log_box.see(tk.END)
                self.log_box.configure(state="disabled")
        except queue.Empty:
            pass
        self.root.after(100, self._poll_logs)

    def _render_current_preview(self, point_cloud_path: str) -> str:
        render_data = self._get_render_data(point_cloud_path)
        artifacts_dir = self._analysis_artifacts_dir(point_cloud_path)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        preview = render_point_cloud_view(
            render_data=render_data,
            output_path=str(artifacts_dir / "point_cloud_preview.png"),
            title="Preview",
            yaw_deg=float(self.preview_yaw_var.get()),
            pitch_deg=float(self.preview_pitch_var.get()),
            width=WORKBENCH_PREVIEW_SIZE,
            height=WORKBENCH_PREVIEW_SIZE,
            log=self._log,
        )
        self.analysis_point_cloud_var.set(point_cloud_path)
        self.preview_status_var.set(
            f"当前预览角度：yaw {self.preview_yaw_var.get():.0f}°, pitch {self.preview_pitch_var.get():.0f}°。"
        )
        return preview.image_path

    def _prepare_views_and_maybe_analyze(
        self,
        *,
        point_cloud_path: str,
        run_openai: bool,
        prompt: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        detail: str | None = None,
    ) -> None:
        artifacts_dir = self._analysis_artifacts_dir(point_cloud_path)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        render_data = self._get_render_data(point_cloud_path)
        preview_path = self._render_current_preview(point_cloud_path)
        view_set = render_point_cloud_view_set(
            render_data=render_data,
            output_dir=str(artifacts_dir / "api_views"),
            width=WORKBENCH_API_VIEW_SIZE,
            height=WORKBENCH_API_VIEW_SIZE,
            log=self._log,
        )
        snapshot = render_point_cloud_snapshot(
            render_data=render_data,
            output_path=str(artifacts_dir / "point_cloud_snapshot.png"),
            log=self._log,
        )
        self.root.after(0, lambda: self.analysis_snapshot_var.set(snapshot.image_path))
        self.root.after(0, lambda: self._set_point_cloud_preview(preview_path))
        self.root.after(
            0,
            lambda names=list(view_set.view_names), paths=list(view_set.image_paths): self._set_api_view_previews(names, paths),
        )
        self.root.after(0, lambda: self.analysis_point_cloud_var.set(point_cloud_path))
        self.root.after(0, lambda: self.notebook.select(self.analysis_tab))

        if not run_openai:
            self.root.after(
                0,
                lambda: self._set_analysis_output(
                    "四张 API 视图已生成：\n"
                    + "\n".join(f"- {path}" for path in view_set.image_paths)
                    + f"\n\n四视图总览：\n{snapshot.image_path}\n\n可继续点击“四图发给 API”。"
                ),
            )
            return

        result = analyze_images_with_openai(
            image_paths=view_set.image_paths,
            prompt=prompt or DEFAULT_ANALYSIS_PROMPT,
            api_key=api_key,
            base_url=base_url,
            model=model or DEFAULT_OPENAI_MODEL,
            detail=detail or "low",
            response_path=str(artifacts_dir / "openai_response.json"),
            log=self._log,
        )
        self.root.after(0, lambda: self.analysis_response_var.set(result.response_path))
        self.root.after(
            0,
            lambda: self._set_analysis_output(
                "本次上传给 API 的视图：\n"
                + "\n".join(f"- {path}" for path in result.image_paths)
                + "\n\nAPI 回答：\n"
                + result.text
            ),
        )

    def run_pipeline_thread(self, *, analyze_after: bool) -> None:
        if self.running:
            return

        work_dir = self.work_dir_var.get().strip()
        if not work_dir:
            messagebox.showerror("参数错误", "请先选择工作目录（输出目录）。如果使用视频输入，程序会在其中生成 images/。")
            return
        try:
            (
                video_path,
                video_sample_fps,
                video_max_frames,
                video_blur_threshold,
                video_dedupe_threshold,
                video_min_gap_sec,
            ) = self._get_video_params()
            mode = self.mode_var.get().strip()
            if mode == MODE_DENSE_ONLY and video_path:
                raise ValueError("仅 Dense 模式不能使用视频输入；请改为全流程或仅 DIM。")
            if analyze_after and mode == MODE_DIM_ONLY:
                raise ValueError("仅 DIM 模式不会生成点云；请改用全流程或仅 Dense。")
            cfg = PipelineConfig(
                work_dir=work_dir,
                pipeline=self.pipeline_var.get().strip() or "superpoint+lightglue",
                colmap_bin=self.colmap_var.get().strip() or "colmap",
                dense_dir=self.dense_dir_var.get().strip() or None,
                gpu=self._str_to_int(self.gpu_var.get()),
                patch_match_gpu=self._str_to_int(self.pm_gpu_var.get()),
                skip_dim=self.skip_dim_var.get(),
                overwrite=self.overwrite_var.get(),
                use_dim_env=self.use_dim_env_var.get(),
                dim_quality=self.dim_quality_var.get().strip() or "medium",
                dim_camera_model=self.dim_camera_model_var.get().strip() or "simple-radial",
                dim_single_camera=not self.dim_multi_camera_var.get(),
                geom_verification=not self.skip_geom_verify_var.get(),
                video_path=video_path,
                video_sample_fps=video_sample_fps,
                video_max_frames=video_max_frames,
                video_blur_threshold=video_blur_threshold,
                video_dedupe_threshold=video_dedupe_threshold,
                video_min_gap_sec=video_min_gap_sec,
            )
            analysis_request = None
            if analyze_after:
                prompt, api_key, base_url, model, detail = self._collect_openai_settings()
                analysis_request = {
                    "prompt": prompt,
                    "api_key": api_key,
                    "base_url": base_url,
                    "model": model,
                    "detail": detail,
                }
        except ValueError as e:
            messagebox.showerror("参数错误", str(e) if str(e) else "GPU 参数必须是整数，抽帧 FPS 必须是数字。")
            return

        self._save_config()
        self.running = True
        self._set_busy(True)
        self._log(f"===== 开始运行 ({mode}) =====" + (" + OpenAI 分析" if analyze_after else ""))

        def worker() -> None:
            try:
                fused: str | None = None
                if mode == MODE_DIM_ONLY:
                    run_dim_step(cfg, log=self._log)
                    self._log("[DONE] DIM + sparse 已完成（查看 work_dir 下 sparse/ 与 dim_outputs/）")
                elif mode == MODE_DENSE_ONLY:
                    fused = run_colmap_mvs(cfg, log=self._log)
                    self._log(f"[DONE] Dense 点云输出: {fused}")
                else:
                    fused = run_pipeline(cfg, log=self._log)
                    self._log(f"[DONE] Dense 点云输出: {fused}")
                if fused:
                    self.root.after(0, lambda p=fused: self._set_point_cloud_result(p))
                    if analyze_after and analysis_request is not None:
                        self._prepare_views_and_maybe_analyze(
                            point_cloud_path=fused,
                            run_openai=True,
                            prompt=analysis_request["prompt"],
                            api_key=analysis_request["api_key"],
                            base_url=analysis_request["base_url"],
                            model=analysis_request["model"],
                            detail=analysis_request["detail"],
                        )
            except Exception as e:  # noqa: BLE001
                self._log(f"[ERROR] {e}")
            finally:
                self.root.after(0, self._on_finished)

        threading.Thread(target=worker, daemon=True).start()

    def snapshot_point_cloud_thread(self) -> None:
        if self.running:
            return

        try:
            point_cloud_path = self.analysis_point_cloud_var.get().strip()
            if not point_cloud_path:
                raise ValueError("请先提供点云文件，或先运行生成点云。")
        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            return

        self._save_config()
        self.running = True
        self._set_busy(True)
        self._log("===== 生成点云四视图 =====")

        def worker() -> None:
            try:
                self._prepare_views_and_maybe_analyze(point_cloud_path=point_cloud_path, run_openai=False)
            except Exception as e:  # noqa: BLE001
                self._log(f"[ERROR] {e}")
            finally:
                self.root.after(0, self._on_finished)

        threading.Thread(target=worker, daemon=True).start()

    def analyze_point_cloud_thread(self) -> None:
        if self.running:
            return

        try:
            point_cloud_path, prompt, api_key, base_url, model, detail = self._collect_analysis_request()
        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            return

        self._save_config()
        self.running = True
        self._set_busy(True)
        self._log("===== 点云四视图 + OpenAI 分析 =====")

        def worker() -> None:
            try:
                self._prepare_views_and_maybe_analyze(
                    point_cloud_path=point_cloud_path,
                    run_openai=True,
                    prompt=prompt,
                    api_key=api_key,
                    base_url=base_url,
                    model=model,
                    detail=detail,
                )
            except Exception as e:  # noqa: BLE001
                self._log(f"[ERROR] {e}")
            finally:
                self.root.after(0, self._on_finished)

        threading.Thread(target=worker, daemon=True).start()

    def _get_dim_env(self) -> DeepImageMatchingEnv:
        return DeepImageMatchingEnv(log_fn=self._log)

    def _run_dim_wrapper_current_env(self, argv: list[str]) -> None:
        env_vars = os.environ.copy()
        gpu = self._str_to_int(self.gpu_var.get())
        if gpu is not None:
            env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [sys.executable, "-m", "uav_pipeline.dim_wrapper", *argv]
        run_cmd(cmd, log=self._log, env=env_vars)

    def _get_test_params(self) -> tuple[str, int | None, str]:
        pipelines = self.test_pipelines_var.get().strip() or "all"
        quality = self.test_quality_var.get().strip() or "lowest"
        try:
            raw = (self.test_max_images_var.get() or "").strip()
            max_images = int(raw) if raw else None
        except ValueError as e:
            raise ValueError("限制图片数必须是整数或留空") from e
        return pipelines, max_images, quality

    def _get_benchmark_params(self) -> tuple[bool, float]:
        if not self.benchmark_var.get():
            return False, 0.2
        try:
            interval = float((self.benchmark_interval_var.get() or "").strip() or "0.2")
        except ValueError as e:
            raise ValueError("benchmark 采样间隔必须是数字") from e
        return True, max(0.05, interval)

    def fill_recommended_pipelines(self) -> None:
        """
        Fill in a curated list of pipelines that are commonly useful for UAV SfM.
        You can still edit the text field afterwards.
        """
        self.test_pipelines_var.set(TEST_PIPELINE_RECOMMENDED)
        self.test_max_images_var.set("")
        self.test_quality_var.set("low")

    def list_pipelines_thread(self) -> None:
        if self.running:
            return

        self.running = True
        self._set_busy(True)
        self._log("===== 列出 DIM pipelines =====")

        def worker() -> None:
            try:
                if self.use_dim_env_var.get():
                    self._get_dim_env().list_pipelines()
                else:
                    self._run_dim_wrapper_current_env(["--list_pipelines"])
            except Exception as e:  # noqa: BLE001
                self._log(f"[ERROR] {e}")
            finally:
                self.root.after(0, self._on_finished)

        threading.Thread(target=worker, daemon=True).start()

    def probe_pipelines_thread(self) -> None:
        if self.running:
            return

        work_dir = self.work_dir_var.get().strip()
        if not work_dir:
            messagebox.showerror("参数错误", "请先选择工作目录（输出目录）。如果使用视频输入，程序会在其中生成 images/。")
            return

        try:
            pipelines, _max_images, quality = self._get_test_params()
            (
                video_path,
                video_sample_fps,
                video_max_frames,
                video_blur_threshold,
                video_dedupe_threshold,
                video_min_gap_sec,
            ) = self._get_video_params()
        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            return

        self._save_config()
        self.running = True
        self._set_busy(True)
        self._log("===== Probe DIM pipelines（仅初始化，不跑匹配） =====")

        def worker() -> None:
            try:
                prepare_work_dir_from_video(
                    work_dir=work_dir,
                    video_path=video_path,
                    sample_fps=video_sample_fps,
                    max_frames=video_max_frames,
                    blur_threshold=video_blur_threshold,
                    dedupe_threshold=video_dedupe_threshold,
                    min_gap_sec=video_min_gap_sec,
                    overwrite=self.overwrite_var.get(),
                    log=self._log,
                )
                if self.use_dim_env_var.get():
                    self._get_dim_env().probe_pipelines(
                        scene_dir=work_dir,
                        pipelines=pipelines,
                        quality=quality,
                        gpu=self._str_to_int(self.gpu_var.get()),
                    )
                else:
                    self._run_dim_wrapper_current_env(
                        [
                            "--dir",
                            work_dir,
                            "--pipelines",
                            pipelines,
                            "--quality",
                            quality,
                            "--probe_pipelines",
                            "--print_summary",
                        ]
                    )
            except Exception as e:  # noqa: BLE001
                self._log(f"[ERROR] {e}")
            finally:
                self.root.after(0, self._on_finished)

        threading.Thread(target=worker, daemon=True).start()

    def test_pipelines_thread(self) -> None:
        if self.running:
            return

        work_dir = self.work_dir_var.get().strip()
        if not work_dir:
            messagebox.showerror("参数错误", "请先选择工作目录（输出目录）。如果使用视频输入，程序会在其中生成 images/。")
            return

        try:
            pipelines, max_images, quality = self._get_test_params()
            do_bench, bench_interval = self._get_benchmark_params()
            (
                video_path,
                video_sample_fps,
                video_max_frames,
                video_blur_threshold,
                video_dedupe_threshold,
                video_min_gap_sec,
            ) = self._get_video_params()
        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            return

        self._save_config()
        self.running = True
        self._set_busy(True)
        run_dense = self.test_run_dense_var.get()
        self._log("===== 跑 DIM pipelines 测试 =====" + (" + Dense" if run_dense else ""))

        def worker() -> None:
            try:
                colmap_bin = self.colmap_var.get().strip() or "colmap"
                prepare_work_dir_from_video(
                    work_dir=work_dir,
                    video_path=video_path,
                    sample_fps=video_sample_fps,
                    max_frames=video_max_frames,
                    blur_threshold=video_blur_threshold,
                    dedupe_threshold=video_dedupe_threshold,
                    min_gap_sec=video_min_gap_sec,
                    overwrite=self.overwrite_var.get(),
                    log=self._log,
                )
                if self.use_dim_env_var.get():
                    self._get_dim_env().test_pipelines(
                        scene_dir=work_dir,
                        pipelines=pipelines,
                        output_dir=None,
                        max_images=max_images,
                        quality=quality,
                        benchmark=do_bench,
                        benchmark_interval=bench_interval,
                        overwrite=self.overwrite_var.get(),
                        single_camera=not self.dim_multi_camera_var.get(),
                        camera_model=self.dim_camera_model_var.get().strip() or "simple-radial",
                        run_dense=run_dense,
                        colmap_bin=colmap_bin,
                        patch_match_gpu=self._str_to_int(self.pm_gpu_var.get()),
                        geom_verification=not self.skip_geom_verify_var.get(),
                        gpu=self._str_to_int(self.gpu_var.get()),
                    )
                else:
                    argv = [
                        "--dir",
                        work_dir,
                        "--pipelines",
                        pipelines,
                        "--quality",
                        quality,
                        "--camera_model",
                        self.dim_camera_model_var.get().strip() or "simple-radial",
                        "--print_summary",
                    ]
                    if do_bench:
                        argv.append("--benchmark")
                        argv += ["--benchmark_interval", str(bench_interval)]
                    if max_images is not None:
                        argv += ["--max_images", str(max_images)]
                    if run_dense:
                        argv.append("--run_dense")
                        argv += ["--colmap_bin", colmap_bin]
                        pm_gpu = self._str_to_int(self.pm_gpu_var.get())
                        if pm_gpu is not None:
                            argv += ["--patch_match_gpu", str(pm_gpu)]
                        if self.skip_geom_verify_var.get():
                            argv.append("--skip_geom_verification")
                    if self.overwrite_var.get():
                        argv.append("--overwrite")
                    if self.dim_multi_camera_var.get():
                        argv.append("--multi_camera")
                    self._run_dim_wrapper_current_env(argv)
            except Exception as e:  # noqa: BLE001
                self._log(f"[ERROR] {e}")
            finally:
                self.root.after(0, self._on_finished)

        threading.Thread(target=worker, daemon=True).start()

    def _on_finished(self) -> None:
        self.running = False
        self._set_busy(False)
        self._log("===== 结束 =====")

    def _apply_saved_config(self) -> None:
        cfg = self._saved_config or {}
        run = cfg.get("run", {})
        test = cfg.get("test", {})
        analysis = cfg.get("analysis", {})

        def _set_str(var: tk.StringVar, value: object) -> None:
            if value is not None:
                var.set(str(value))

        def _set_bool(var: tk.BooleanVar, value: object) -> None:
            if value is not None:
                var.set(bool(value))

        _set_str(self.work_dir_var, run.get("work_dir"))
        _set_str(self.video_var, run.get("video_path"))
        _set_str(self.video_sample_fps_var, run.get("video_sample_fps"))
        _set_str(self.video_max_frames_var, run.get("video_max_frames"))
        _set_str(self.video_blur_threshold_var, run.get("video_blur_threshold"))
        _set_str(self.video_dedupe_threshold_var, run.get("video_dedupe_threshold"))
        _set_str(self.video_min_gap_sec_var, run.get("video_min_gap_sec"))
        _set_str(self.colmap_var, run.get("colmap_bin"))
        _set_str(self.pipeline_var, run.get("pipeline"))
        _set_str(self.dense_dir_var, run.get("dense_dir"))
        _set_str(self.gpu_var, run.get("gpu"))
        _set_str(self.pm_gpu_var, run.get("patch_match_gpu"))
        _set_str(self.dim_quality_var, run.get("dim_quality"))
        _set_str(self.dim_camera_model_var, run.get("dim_camera_model"))
        _set_str(self.mode_var, run.get("mode"))

        _set_bool(self.use_dim_env_var, run.get("use_dim_env"))
        _set_bool(self.skip_dim_var, run.get("skip_dim"))
        _set_bool(self.overwrite_var, run.get("overwrite"))
        _set_bool(self.dim_multi_camera_var, run.get("dim_multi_camera"))
        _set_bool(self.skip_geom_verify_var, run.get("skip_geom_verification"))

        _set_str(self.test_pipelines_var, test.get("pipelines"))
        _set_str(self.test_max_images_var, test.get("max_images"))
        _set_str(self.test_quality_var, test.get("quality"))
        _set_str(self.benchmark_interval_var, test.get("benchmark_interval"))
        _set_bool(self.benchmark_var, test.get("benchmark"))
        _set_bool(self.test_run_dense_var, test.get("run_dense"))

        _set_str(self.analysis_point_cloud_var, analysis.get("point_cloud_path"))
        _set_str(self.analysis_snapshot_var, analysis.get("snapshot_path"))
        _set_str(self.analysis_response_var, analysis.get("response_path"))
        self.openai_api_key_var.set("")
        _set_str(self.openai_base_url_var, analysis.get("openai_base_url"))
        _set_str(self.openai_model_var, analysis.get("openai_model"))
        _set_str(self.openai_detail_var, analysis.get("openai_detail"))
        if analysis.get("preview_yaw") is not None:
            self.preview_yaw_var.set(float(analysis.get("preview_yaw")))
        if analysis.get("preview_pitch") is not None:
            self.preview_pitch_var.set(float(analysis.get("preview_pitch")))
        self._set_prompt_text(str(analysis.get("prompt") or DEFAULT_ANALYSIS_PROMPT))
        self._update_preview_angle_labels()
        preview_path = None
        point_cloud_path = self.analysis_point_cloud_var.get().strip()
        if point_cloud_path:
            artifacts_dir = self._analysis_artifacts_dir(point_cloud_path)
            candidate_preview = artifacts_dir / "point_cloud_preview.png"
            if candidate_preview.exists():
                preview_path = str(candidate_preview)
            api_dir = artifacts_dir / "api_views"
            existing_api_names: list[str] = []
            existing_api_paths: list[str] = []
            for view_name, _, _ in DEFAULT_POINT_CLOUD_API_VIEWS:
                candidate = api_dir / f"{view_name.lower()}.png"
                if candidate.exists():
                    existing_api_names.append(view_name)
                    existing_api_paths.append(str(candidate))
            if existing_api_paths:
                self._set_api_view_previews(existing_api_names, existing_api_paths)
            else:
                self._clear_api_view_previews()
        else:
            self._clear_api_view_previews()
        self._set_point_cloud_preview(preview_path)
        self._sync_mode_ui()

    def _collect_gui_config(self) -> dict[str, dict[str, object]]:
        run = {
            "work_dir": self.work_dir_var.get().strip(),
            "video_path": self.video_var.get().strip(),
            "video_sample_fps": self.video_sample_fps_var.get().strip(),
            "video_max_frames": self.video_max_frames_var.get().strip(),
            "video_blur_threshold": self.video_blur_threshold_var.get().strip(),
            "video_dedupe_threshold": self.video_dedupe_threshold_var.get().strip(),
            "video_min_gap_sec": self.video_min_gap_sec_var.get().strip(),
            "colmap_bin": self.colmap_var.get().strip(),
            "pipeline": self.pipeline_var.get().strip(),
            "dense_dir": self.dense_dir_var.get().strip(),
            "gpu": self.gpu_var.get().strip(),
            "patch_match_gpu": self.pm_gpu_var.get().strip(),
            "dim_quality": self.dim_quality_var.get().strip(),
            "dim_camera_model": self.dim_camera_model_var.get().strip(),
            "mode": self.mode_var.get().strip(),
            "use_dim_env": self.use_dim_env_var.get(),
            "skip_dim": self.skip_dim_var.get(),
            "overwrite": self.overwrite_var.get(),
            "dim_multi_camera": self.dim_multi_camera_var.get(),
            "skip_geom_verification": self.skip_geom_verify_var.get(),
        }
        test = {
            "pipelines": self.test_pipelines_var.get().strip(),
            "max_images": self.test_max_images_var.get().strip(),
            "quality": self.test_quality_var.get().strip(),
            "benchmark": self.benchmark_var.get(),
            "benchmark_interval": self.benchmark_interval_var.get().strip(),
            "run_dense": self.test_run_dense_var.get(),
        }
        analysis = {
            "point_cloud_path": self.analysis_point_cloud_var.get().strip(),
            "snapshot_path": self.analysis_snapshot_var.get().strip(),
            "response_path": self.analysis_response_var.get().strip(),
            "openai_api_key": "",
            "openai_base_url": self.openai_base_url_var.get().strip()
            or os.environ.get(OPENAI_BASE_URL_ENV, DEFAULT_OPENAI_BASE_URL),
            "openai_model": self.openai_model_var.get().strip(),
            "openai_detail": self.openai_detail_var.get().strip(),
            "preview_yaw": f"{self.preview_yaw_var.get():.2f}",
            "preview_pitch": f"{self.preview_pitch_var.get():.2f}",
            "prompt": self._get_prompt_text() or DEFAULT_ANALYSIS_PROMPT,
        }
        return {"run": run, "test": test, "analysis": analysis}

    def _save_config(self) -> None:
        update_section("gui", self._collect_gui_config())

    def _on_close(self) -> None:
        self._save_config()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    PipelineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
