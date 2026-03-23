from __future__ import annotations

import os
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from .colmap_runtime import detect_colmap_binary, ensure_colmap_binary
from .meshlab_runtime import detect_meshlab_binary, ensure_meshlab_binary, open_point_cloud_in_meshlab
from .openai_analysis import (
    DEFAULT_OPENAI_BASE_URL,
    OPENAI_API_KEY_ENV,
    OPENAI_BASE_URL_ENV,
    OPENAI_MODEL_ENV,
    analyze_images_with_openai,
    load_openai_runtime_defaults,
)
from .pipeline import PipelineConfig, run_pipeline
from .point_cloud_snapshot import (
    DEFAULT_POINT_CLOUD_API_VIEWS,
    POINT_CLOUD_RENDER_STYLES,
    PointCloudRenderData,
    prepare_point_cloud_render_data,
    render_point_cloud_snapshot,
    render_point_cloud_view,
    render_point_cloud_view_set,
)
from .user_config import load_section, update_section

DIM_QUALITY_OPTIONS = ("highest", "high", "medium", "low", "lowest")
OPENAI_IMAGE_DETAIL_OPTIONS = ("low", "high", "auto")
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_ANALYSIS_PROMPT = (
    "请用中文综合分析这组点云四视图：\n"
    "1. 描述可见建筑的大致分布、高低和密集程度；\n"
    "2. 描述空旷区域的大致位置、范围和开阔程度；\n"
    "3. 根据建筑与空旷区域的组合，给出这个地方可能是什么类型区域的一个大概猜测；\n"
    "4. 说明这个区域更像是居住、工业、校园、商业、仓储、乡村聚落还是其他什么类型，并给出简短理由。"
)
LEGACY_ANALYSIS_PROMPTS = {
    "请用中文分析这张点云截图：\n"
    "1. 这更像什么场景或结构；\n"
    "2. 重建是否完整，哪里缺失明显；\n"
    "3. 噪点或伪影主要集中在哪；\n"
    "4. 如果要继续采集，请给出 3 条简短建议。",
    "请用中文综合分析这组点云四视图：\n"
    "1. 这更像什么场景或结构；\n"
    "2. 重建是否完整，哪里缺失明显；\n"
    "3. 噪点或伪影主要集中在哪；\n"
    "4. 如果要继续采集，请给出 3 条简短建议。",
}
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
RENDER_STYLE_LABELS = {
    "点状": "points",
    "面状": "surface",
}
PREVIEW_SIZE = 520
API_VIEW_SIZE = 250
PREVIEW_MIN_ZOOM = 0.35
PREVIEW_MAX_ZOOM = 6.0
PREVIEW_MAX_PAN = 2.5


def _detect_colmap_bin() -> str:
    detected = detect_colmap_binary()
    return detected.binary_path if detected is not None else "colmap"


class WorkbenchGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("UAV Point Cloud Workbench")
        self.root.minsize(1480, 860)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.running = False
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.preview_photo: tk.PhotoImage | None = None
        self.api_view_photos: dict[str, tk.PhotoImage] = {}
        self.api_view_paths: list[str] = []
        self.preview_pan_x = 0.0
        self.preview_pan_y = 0.0
        self.preview_zoom = 1.0
        self._preview_refresh_after_id: str | None = None
        self._preview_refresh_running = False
        self._preview_refresh_pending = False
        self._preview_drag_mode: str | None = None
        self._preview_drag_origin: tuple[int, int, float, float, float, float] | None = None
        self._render_data_cache: PointCloudRenderData | None = None
        self._render_data_cache_path: str | None = None
        self._meshlab_binary_path = ""
        self._saved_config = load_section("workbench_gui")
        self._runtime_openai_defaults = load_openai_runtime_defaults()

        self._style()
        self._build_vars()
        self._build_layout()
        self._apply_saved_config()
        self._apply_runtime_constraints()
        self._poll_logs()

    def _style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("Section.TLabelframe.Label", font=("Segoe UI", 10, "bold"))

    def _build_vars(self) -> None:
        self.work_dir_var = tk.StringVar()
        self.video_var = tk.StringVar()
        self.point_cloud_var = tk.StringVar()
        self.colmap_var = tk.StringVar(value=_detect_colmap_bin())
        self.pipeline_var = tk.StringVar(value="aliked+lightglue")
        self.dim_quality_var = tk.StringVar(value="medium")
        self.render_style_label_var = tk.StringVar(value="面状")
        self.video_sample_fps_var = tk.StringVar(value="1.0")
        self.video_max_frames_var = tk.StringVar(value="24")
        self.video_blur_threshold_var = tk.StringVar(value="2000")
        self.video_dedupe_threshold_var = tk.StringVar(value="4.0")
        self.video_min_gap_sec_var = tk.StringVar(value="1.0")
        self.overwrite_var = tk.BooleanVar(value=True)
        self.use_dim_env_var = tk.BooleanVar(value=True)

        self.preview_yaw_var = tk.DoubleVar(value=35.0)
        self.preview_pitch_var = tk.DoubleVar(value=-25.0)
        self.preview_status_var = tk.StringVar(value="等待点云。")
        self.api_view_status_var = tk.StringVar(value="尚未生成四张 API 图片。")

        self.openai_base_url_var = tk.StringVar(
            value=self._runtime_openai_defaults.get(OPENAI_BASE_URL_ENV, DEFAULT_OPENAI_BASE_URL)
        )
        self.openai_model_var = tk.StringVar(
            value=self._runtime_openai_defaults.get(OPENAI_MODEL_ENV, DEFAULT_OPENAI_MODEL)
        )
        self.openai_detail_var = tk.StringVar(value="low")
        self.response_path_var = tk.StringVar()
        self.snapshot_path_var = tk.StringVar()

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(outer)
        header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header, text="UAV Point Cloud Workbench", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(
            header,
            text="视频 -> 点云 -> 可旋转预览 -> 四视图 -> API 输出",
        ).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Button(header, text="清空日志", command=self.clear_log).pack(side=tk.RIGHT)

        workspace = ttk.Frame(outer)
        workspace.pack(fill=tk.BOTH, expand=True)
        workspace.grid_columnconfigure(0, weight=2)
        workspace.grid_columnconfigure(1, weight=3)
        workspace.grid_columnconfigure(2, weight=3)
        workspace.grid_columnconfigure(3, weight=3)
        workspace.grid_rowconfigure(0, weight=1)

        self._build_input_column(workspace)
        self._build_preview_column(workspace)
        self._build_api_views_column(workspace)
        self._build_output_column(workspace)

        log_frame = ttk.LabelFrame(outer, text="日志", style="Section.TLabelframe", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=False, pady=(10, 0))
        self.log_box = ScrolledText(log_frame, height=12, wrap="word", state="disabled")
        self.log_box.pack(fill=tk.BOTH, expand=True)

    def _build_input_column(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="1. 输入与运行", style="Section.TLabelframe", padding=10)
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.work_dir_entry = self._grid_labeled_entry(frame, "工作目录:", self.work_dir_var, row=0, browse="dir")
        self.video_entry = self._grid_labeled_entry(frame, "视频文件:", self.video_var, row=1, browse="file")
        self.point_cloud_entry = self._grid_labeled_entry(frame, "点云文件 (.ply):", self.point_cloud_var, row=2, browse="file")
        self.colmap_entry = self._grid_labeled_entry(frame, "COLMAP:", self.colmap_var, row=3, browse="file")
        self.colmap_auto_btn = ttk.Button(frame, text="自动检测/下载 COLMAP", command=self.configure_colmap_thread)
        self.colmap_auto_btn.grid(row=4, column=1, columnspan=2, sticky="we", pady=(0, 6))
        self.pipeline_combo = self._grid_labeled_combo(
            frame,
            "DIM pipeline:",
            self.pipeline_var,
            values=PIPELINE_PRESETS,
            row=5,
            editable=True,
        )
        self.dim_quality_combo = self._grid_labeled_combo(
            frame,
            "DIM quality:",
            self.dim_quality_var,
            values=DIM_QUALITY_OPTIONS,
            row=6,
        )
        self.render_style_combo = self._grid_labeled_combo(
            frame,
            "显示风格:",
            self.render_style_label_var,
            values=tuple(RENDER_STYLE_LABELS.keys()),
            row=7,
        )
        self.video_fps_entry = self._grid_labeled_entry(frame, "抽帧 FPS:", self.video_sample_fps_var, row=8)
        self.video_max_frames_entry = self._grid_labeled_entry(frame, "最大帧数:", self.video_max_frames_var, row=9)
        self.video_blur_entry = self._grid_labeled_entry(frame, "清晰度阈值:", self.video_blur_threshold_var, row=10)
        self.video_dedupe_entry = self._grid_labeled_entry(frame, "去重阈值:", self.video_dedupe_threshold_var, row=11)
        self.video_min_gap_entry = self._grid_labeled_entry(frame, "最小间隔秒:", self.video_min_gap_sec_var, row=12)

        flags = ttk.Frame(frame)
        flags.grid(row=13, column=0, columnspan=3, sticky="w", pady=(6, 6))
        self.overwrite_check = ttk.Checkbutton(flags, text="覆盖输出", variable=self.overwrite_var)
        self.overwrite_check.pack(side=tk.LEFT)
        self.use_dim_env_check = ttk.Checkbutton(flags, text="Use managed DIM env", variable=self.use_dim_env_var)
        self.use_dim_env_check.pack(side=tk.LEFT, padx=(12, 0))

        actions = ttk.Frame(frame)
        actions.grid(row=14, column=0, columnspan=3, sticky="we", pady=(4, 6))
        for column in range(5):
            actions.grid_columnconfigure(column, weight=1)
        self.run_btn = ttk.Button(actions, text="生成点云", command=self.run_pipeline_thread)
        self.run_btn.grid(row=0, column=0, sticky="we", padx=(0, 4))
        self.prepare_views_btn = ttk.Button(actions, text="生成四图", command=self.prepare_views_thread)
        self.prepare_views_btn.grid(row=0, column=1, sticky="we", padx=4)
        self.run_and_analyze_btn = ttk.Button(actions, text="生成并分析", command=self.run_and_analyze_thread)
        self.run_and_analyze_btn.grid(row=0, column=2, sticky="we", padx=4)
        self.analyze_current_btn = ttk.Button(actions, text="分析当前点云", command=self.analyze_current_point_cloud_thread)
        self.analyze_current_btn.grid(row=0, column=3, sticky="we", padx=4)
        self.open_meshlab_btn = ttk.Button(actions, text="MeshLab 打开点云", command=self.open_point_cloud_in_meshlab_thread)
        self.open_meshlab_btn.grid(row=0, column=4, sticky="we", padx=(4, 0))

        self.input_hint = ttk.Label(
            frame,
            text=(
                "如果你已经有点云，只需要选择 `.ply` 再点“生成四图”或“生成并分析”。\n"
                "如果从视频开始，工作目录会作为输出目录，图片/稀疏/稠密/analysis 都写进去。\n"
                "Windows 下如果没装 COLMAP，可以直接点上面的“自动检测/下载 COLMAP”，GUI 会优先探测本机安装，"
                "找不到时再下载官方 Windows 包到应用内部目录并自动填好路径。\n"
                "如果已经有点云，可以直接点“MeshLab 打开点云”；没装 MeshLab 时，工作台会先自动下载官方 Windows 版。\n"
                "如果点云已经生成好，不用再重跑，只要点“分析当前点云”就会直接截图并调用 API。\n"
                "“面状”只影响显示观感，不会改动真实点云或伪造 mesh 文件。"
            ),
            wraplength=280,
            justify="left",
        )
        self.input_hint.grid(row=15, column=0, columnspan=3, sticky="we")

    def _build_preview_column(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="2. 点云预览（鼠标可交互）", style="Section.TLabelframe", padding=10)
        frame.grid(row=0, column=1, sticky="nsew", padx=8)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=1)

        self.preview_label = ttk.Label(frame, text="尚未生成点云预览。", anchor="center")
        self.preview_label.grid(row=0, column=0, sticky="nsew")
        self.preview_label.configure(cursor="crosshair")

        controls = ttk.Frame(frame)
        controls.grid(row=1, column=0, sticky="we", pady=(10, 0))
        controls.grid_columnconfigure(1, weight=1)
        ttk.Label(controls, text="Yaw").grid(row=0, column=0, sticky="w")
        self.preview_yaw_scale = ttk.Scale(controls, from_=-180, to=180, variable=self.preview_yaw_var)
        self.preview_yaw_scale.grid(row=0, column=1, sticky="we", padx=(8, 8))
        self.preview_yaw_value = ttk.Label(controls, text="35°")
        self.preview_yaw_value.grid(row=0, column=2, sticky="e")
        ttk.Label(controls, text="Pitch").grid(row=1, column=0, sticky="w")
        self.preview_pitch_scale = ttk.Scale(controls, from_=-90, to=90, variable=self.preview_pitch_var)
        self.preview_pitch_scale.grid(row=1, column=1, sticky="we", padx=(8, 8))
        self.preview_pitch_value = ttk.Label(controls, text="-25°")
        self.preview_pitch_value.grid(row=1, column=2, sticky="e")

        buttons = ttk.Frame(frame)
        buttons.grid(row=2, column=0, sticky="we", pady=(8, 0))
        for column in range(5):
            buttons.grid_columnconfigure(column, weight=1)
        self.preview_left_btn = ttk.Button(buttons, text="左转", command=lambda: self._nudge_preview(yaw_delta=-15))
        self.preview_left_btn.grid(row=0, column=0, sticky="we", padx=(0, 4))
        self.preview_right_btn = ttk.Button(buttons, text="右转", command=lambda: self._nudge_preview(yaw_delta=15))
        self.preview_right_btn.grid(row=0, column=1, sticky="we", padx=4)
        self.preview_up_btn = ttk.Button(buttons, text="抬高", command=lambda: self._nudge_preview(pitch_delta=10))
        self.preview_up_btn.grid(row=0, column=2, sticky="we", padx=4)
        self.preview_down_btn = ttk.Button(buttons, text="压低", command=lambda: self._nudge_preview(pitch_delta=-10))
        self.preview_down_btn.grid(row=0, column=3, sticky="we", padx=4)
        self.preview_refresh_btn = ttk.Button(buttons, text="刷新", command=self.refresh_preview_thread)
        self.preview_refresh_btn.grid(row=0, column=4, sticky="we", padx=(4, 0))
        self.preview_reset_btn = ttk.Button(frame, text="重置视角", command=self.reset_preview_thread)
        self.preview_reset_btn.grid(row=3, column=0, sticky="we", pady=(6, 0))

        self.preview_status_label = ttk.Label(frame, textvariable=self.preview_status_var, wraplength=420, justify="left")
        self.preview_status_label.grid(row=4, column=0, sticky="we", pady=(8, 0))

        self.preview_yaw_scale.bind("<ButtonRelease-1>", lambda _event: self.refresh_preview_thread())
        self.preview_pitch_scale.bind("<ButtonRelease-1>", lambda _event: self.refresh_preview_thread())
        self.preview_label.bind("<ButtonPress-1>", self._on_preview_rotate_press)
        self.preview_label.bind("<B1-Motion>", self._on_preview_rotate_drag)
        self.preview_label.bind("<ButtonRelease-1>", self._on_preview_drag_release)
        self.preview_label.bind("<ButtonPress-3>", self._on_preview_pan_press)
        self.preview_label.bind("<B3-Motion>", self._on_preview_pan_drag)
        self.preview_label.bind("<ButtonRelease-3>", self._on_preview_drag_release)
        self.preview_label.bind("<MouseWheel>", self._on_preview_mousewheel)

    def _build_api_views_column(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="3. 发给 API 的四张视图", style="Section.TLabelframe", padding=10)
        frame.grid(row=0, column=2, sticky="nsew", padx=8)
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        self.api_view_labels: dict[str, ttk.Label] = {}
        for index, (view_name, _, _) in enumerate(DEFAULT_POINT_CLOUD_API_VIEWS):
            holder = ttk.LabelFrame(frame, text=view_name, padding=6)
            holder.grid(row=index // 2, column=index % 2, sticky="nsew", padx=4, pady=4)
            holder.grid_columnconfigure(0, weight=1)
            holder.grid_rowconfigure(0, weight=1)
            label = ttk.Label(holder, text=f"{view_name}\n尚未生成", anchor="center")
            label.grid(row=0, column=0, sticky="nsew")
            self.api_view_labels[view_name] = label

        self.api_view_status_label = ttk.Label(frame, textvariable=self.api_view_status_var, wraplength=420, justify="left")
        self.api_view_status_label.grid(row=2, column=0, columnspan=2, sticky="we", pady=(8, 0))

    def _build_output_column(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="4. API 输出", style="Section.TLabelframe", padding=10)
        frame.grid(row=0, column=3, sticky="nsew", padx=(8, 0))
        frame.grid_columnconfigure(1, weight=1)
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_rowconfigure(4, weight=2)

        ttk.Label(frame, text="API 配置:").grid(row=0, column=0, sticky="nw", pady=4)
        ttk.Label(
            frame,
            text="固定使用运行时配置：环境变量或源码目录 openai.env",
            wraplength=360,
            justify="left",
        ).grid(row=0, column=1, columnspan=2, sticky="w", pady=4, padx=(8, 0))

        ttk.Label(frame, text="分析提示词:").grid(row=1, column=0, sticky="nw", pady=4)
        self.prompt_box = ScrolledText(frame, height=9, wrap="word")
        self.prompt_box.grid(row=1, column=1, columnspan=2, sticky="nsew", pady=4, padx=(8, 0))

        self.output_hint = ttk.Label(
            frame,
            text=(
                f"API 固定读取环境变量 {OPENAI_API_KEY_ENV} 或源码目录 `openai.env`；GUI 不再手动覆盖 key。\n"
                "模型、Base URL、图像细节等高级项仍然走运行时配置或内置默认值。"
            ),
            wraplength=360,
            justify="left",
        )
        self.output_hint.grid(row=2, column=0, columnspan=3, sticky="we", pady=(8, 4))

        ttk.Label(frame, text="API 回答:").grid(row=4, column=0, sticky="nw", pady=(8, 0))
        self.result_box = ScrolledText(frame, height=18, wrap="word", state="disabled")
        self.result_box.grid(row=4, column=1, columnspan=2, sticky="nsew", pady=(8, 0), padx=(8, 0))

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
        return combo

    def _choose_dir(self, var: tk.StringVar) -> None:
        path = filedialog.askdirectory(title="选择目录")
        if path:
            var.set(path)

    def _choose_file(self, var: tk.StringVar) -> None:
        path = filedialog.askopenfilename(title="选择文件")
        if path:
            var.set(path)

    def clear_log(self) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", tk.END)
        self.log_box.configure(state="disabled")

    def _log(self, text: str) -> None:
        self.log_queue.put(text)

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

    def _set_text(self, widget: ScrolledText, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _set_preview_image(self, image_path: str | None) -> None:
        self.preview_yaw_value.configure(text=f"{self.preview_yaw_var.get():.0f}°")
        self.preview_pitch_value.configure(text=f"{self.preview_pitch_var.get():.0f}°")
        if not image_path:
            self.preview_photo = None
            self.preview_label.configure(text="尚未生成点云预览。", image="")
            return
        try:
            photo = tk.PhotoImage(file=image_path)
        except tk.TclError:
            self.preview_photo = None
            self.preview_label.configure(text=f"预览已生成，但 Tk 无法显示：{image_path}", image="")
            return
        self.preview_photo = photo
        self.preview_label.configure(text="", image=self.preview_photo)
        self._update_preview_status()

    def _preview_status_text(self, *, prefix: str | None = None) -> str:
        status = (
            "左键拖动旋转，右键拖动平移，滚轮缩放。"
            f" 当前视角：yaw {self.preview_yaw_var.get():.0f}°,"
            f" pitch {self.preview_pitch_var.get():.0f}°,"
            f" pan ({self.preview_pan_x:+.2f}, {self.preview_pan_y:+.2f}),"
            f" zoom {self.preview_zoom:.2f}x。"
        )
        return f"{prefix} {status}" if prefix else status

    def _update_preview_status(self, *, prefix: str | None = None) -> None:
        self.preview_status_var.set(self._preview_status_text(prefix=prefix))

    def _clear_api_view_images(self) -> None:
        self.api_view_paths = []
        self.api_view_photos = {}
        for view_name, label in self.api_view_labels.items():
            label.configure(text=f"{view_name}\n尚未生成", image="")
        self.api_view_status_var.set("尚未生成四张 API 图片。")

    def _set_api_view_images(self, image_paths: list[str]) -> None:
        self.api_view_paths = list(image_paths)
        self.api_view_photos = {}
        for view_name, image_path in zip((name for name, _, _ in DEFAULT_POINT_CLOUD_API_VIEWS), image_paths, strict=False):
            label = self.api_view_labels[view_name]
            try:
                photo = tk.PhotoImage(file=image_path)
            except tk.TclError:
                label.configure(text=f"{view_name}\n{image_path}", image="")
                continue
            self.api_view_photos[view_name] = photo
            label.configure(text="", image=photo)
        self.api_view_status_var.set("四张 API 图片已更新。发送时会按 Front / Right / Top / Isometric 顺序上传。")

    def _analysis_artifacts_dir(self, point_cloud_path: str) -> Path:
        point_cloud = Path(point_cloud_path).expanduser().resolve()
        if point_cloud.parent.name == "dense" and point_cloud.parent.parent.exists():
            return point_cloud.parent.parent / "analysis"
        return point_cloud.parent / "analysis"

    def _get_render_data(self, point_cloud_path: str) -> PointCloudRenderData:
        resolved = str(Path(point_cloud_path).expanduser().resolve())
        if self._render_data_cache is not None and self._render_data_cache_path == resolved:
            return self._render_data_cache
        render_data = prepare_point_cloud_render_data(point_cloud_path=resolved)
        self._render_data_cache = render_data
        self._render_data_cache_path = resolved
        return render_data

    def _reset_render_cache(self) -> None:
        self._render_data_cache = None
        self._render_data_cache_path = None

    def _set_point_cloud_path(self, point_cloud_path: str) -> None:
        self.point_cloud_var.set(point_cloud_path)
        self.snapshot_path_var.set("")
        self.response_path_var.set("")
        self._reset_render_cache()
        self._clear_api_view_images()
        self._set_preview_image(None)
        self._update_preview_status(prefix="点云已更新。点击“刷新”或直接拖动预览。")
        self._set_text(self.result_box, "")

    def _ensure_colmap_binary_for_run(self, *, download_if_missing: bool) -> str:
        preferred = self.colmap_var.get().strip()
        result = ensure_colmap_binary(
            preferred_path=preferred,
            download_if_missing=download_if_missing and os.name == "nt",
            log=self._log,
        )
        self.root.after(0, lambda path=result.binary_path: self.colmap_var.set(path))
        self._log(f"[OK] COLMAP ready ({result.source}): {result.binary_path}")
        return result.binary_path

    def _ensure_meshlab_binary_for_run(self, *, download_if_missing: bool) -> str:
        preferred = self._meshlab_binary_path or None
        result = detect_meshlab_binary(preferred)
        if result is None:
            result = ensure_meshlab_binary(
                preferred_path=preferred,
                download_if_missing=download_if_missing and os.name == "nt",
                log=self._log,
            )
        self._meshlab_binary_path = result.binary_path
        self._log(f"[OK] MeshLab ready ({result.source}): {result.binary_path}")
        return result.binary_path

    def _collect_pipeline_config(self) -> PipelineConfig:
        work_dir = self.work_dir_var.get().strip()
        if not work_dir:
            raise ValueError("请先选择工作目录。")
        video_path = self.video_var.get().strip() or None
        if not video_path and not self.point_cloud_var.get().strip():
            raise ValueError("请提供视频文件，或者选择已有点云。")
        try:
            sample_fps = float((self.video_sample_fps_var.get() or "1.0").strip())
            max_frames_raw = (self.video_max_frames_var.get() or "").strip()
            max_frames = int(max_frames_raw) if max_frames_raw else None
            blur_threshold = float((self.video_blur_threshold_var.get() or "0").strip() or "0")
            dedupe_threshold = float((self.video_dedupe_threshold_var.get() or "0").strip() or "0")
            min_gap_sec = float((self.video_min_gap_sec_var.get() or "0").strip() or "0")
        except ValueError as exc:
            raise ValueError("抽帧参数必须是合法数字。") from exc
        colmap_bin = self._ensure_colmap_binary_for_run(download_if_missing=True)
        return PipelineConfig(
            work_dir=work_dir,
            pipeline=self.pipeline_var.get().strip() or "aliked+lightglue",
            colmap_bin=colmap_bin,
            overwrite=self.overwrite_var.get(),
            use_dim_env=True if getattr(sys, "frozen", False) else self.use_dim_env_var.get(),
            dim_quality=self.dim_quality_var.get().strip() or "medium",
            video_path=video_path,
            video_sample_fps=sample_fps,
            video_max_frames=max_frames,
            video_blur_threshold=blur_threshold,
            video_dedupe_threshold=dedupe_threshold,
            video_min_gap_sec=min_gap_sec,
        )

    def _collect_api_settings(self) -> tuple[str, str, str, str, str]:
        prompt = self.prompt_box.get("1.0", tk.END).strip()
        if not prompt:
            raise ValueError("分析提示词不能为空。")
        api_key = ""
        base_url = (
            self._runtime_openai_defaults.get(OPENAI_BASE_URL_ENV)
            or self.openai_base_url_var.get().strip()
            or os.environ.get(OPENAI_BASE_URL_ENV, DEFAULT_OPENAI_BASE_URL)
        )
        model = (
            self._runtime_openai_defaults.get(OPENAI_MODEL_ENV)
            or self.openai_model_var.get().strip()
            or DEFAULT_OPENAI_MODEL
        )
        detail = self.openai_detail_var.get().strip() or "low"
        return prompt, api_key, base_url, model, detail

    def _current_render_style(self) -> str:
        label = self.render_style_label_var.get().strip() or "面状"
        render_style = RENDER_STYLE_LABELS.get(label, "surface")
        if render_style not in POINT_CLOUD_RENDER_STYLES:
            return "surface"
        return render_style

    def _nudge_preview(self, *, yaw_delta: float = 0.0, pitch_delta: float = 0.0) -> None:
        self.preview_yaw_var.set(self.preview_yaw_var.get() + yaw_delta)
        self.preview_pitch_var.set(max(-90.0, min(90.0, self.preview_pitch_var.get() + pitch_delta)))
        self._update_preview_status(prefix="正在刷新预览。")
        self.refresh_preview_thread()

    def _preview_interaction_enabled(self) -> bool:
        return bool(self.point_cloud_var.get().strip()) and not self.running

    def _preview_drag_extent(self) -> tuple[float, float]:
        width = max(self.preview_label.winfo_width(), PREVIEW_SIZE)
        height = max(self.preview_label.winfo_height(), PREVIEW_SIZE)
        return float(width), float(height)

    def _schedule_preview_refresh(self, *, delay_ms: int = 50) -> None:
        if not self._preview_interaction_enabled():
            return
        self._preview_refresh_pending = True
        self._update_preview_status(prefix="正在刷新预览。")
        if self._preview_refresh_running:
            return
        if self._preview_refresh_after_id is not None:
            self.root.after_cancel(self._preview_refresh_after_id)
        self._preview_refresh_after_id = self.root.after(delay_ms, self._start_preview_refresh)

    def _start_preview_refresh(self) -> None:
        self._preview_refresh_after_id = None
        if not self._preview_interaction_enabled():
            self._preview_refresh_pending = False
            return
        if self._preview_refresh_running:
            self._preview_refresh_pending = True
            return
        point_cloud_path = self.point_cloud_var.get().strip()
        if not point_cloud_path:
            self._preview_refresh_pending = False
            return

        self._preview_refresh_pending = False
        self._preview_refresh_running = True

        def worker() -> None:
            try:
                preview_path = self._render_preview(point_cloud_path)
            except Exception as exc:  # noqa: BLE001
                self.root.after(0, lambda message=str(exc): self._on_preview_refresh_failed(message))
                return
            self.root.after(0, lambda path=preview_path: self._on_preview_refresh_ready(path))

        threading.Thread(target=worker, daemon=True).start()

    def _on_preview_refresh_ready(self, image_path: str) -> None:
        self._preview_refresh_running = False
        self._set_preview_image(image_path)
        if self._preview_refresh_pending:
            self._schedule_preview_refresh(delay_ms=10)
            return
        self._update_preview_status()

    def _on_preview_refresh_failed(self, message: str) -> None:
        self._preview_refresh_running = False
        self.preview_status_var.set(f"预览刷新失败：{message}")
        self._log(f"[ERROR] Preview refresh failed: {message}")

    def _on_preview_rotate_press(self, event: tk.Event) -> str:
        if not self._preview_interaction_enabled():
            return "break"
        self._preview_drag_mode = "rotate"
        self._preview_drag_origin = (
            int(event.x),
            int(event.y),
            float(self.preview_yaw_var.get()),
            float(self.preview_pitch_var.get()),
            self.preview_pan_x,
            self.preview_pan_y,
        )
        return "break"

    def _on_preview_rotate_drag(self, event: tk.Event) -> str:
        if self._preview_drag_mode != "rotate" or self._preview_drag_origin is None:
            return "break"
        start_x, start_y, start_yaw, start_pitch, _start_pan_x, _start_pan_y = self._preview_drag_origin
        width, height = self._preview_drag_extent()
        yaw_delta = (float(event.x) - start_x) * (240.0 / max(width, 240.0))
        pitch_delta = (start_y - float(event.y)) * (180.0 / max(height, 240.0))
        self.preview_yaw_var.set(start_yaw + yaw_delta)
        self.preview_pitch_var.set(max(-90.0, min(90.0, start_pitch + pitch_delta)))
        self._schedule_preview_refresh(delay_ms=25)
        return "break"

    def _on_preview_pan_press(self, event: tk.Event) -> str:
        if not self._preview_interaction_enabled():
            return "break"
        self._preview_drag_mode = "pan"
        self._preview_drag_origin = (
            int(event.x),
            int(event.y),
            float(self.preview_yaw_var.get()),
            float(self.preview_pitch_var.get()),
            self.preview_pan_x,
            self.preview_pan_y,
        )
        return "break"

    def _on_preview_pan_drag(self, event: tk.Event) -> str:
        if self._preview_drag_mode != "pan" or self._preview_drag_origin is None:
            return "break"
        start_x, start_y, _start_yaw, _start_pitch, start_pan_x, start_pan_y = self._preview_drag_origin
        width, height = self._preview_drag_extent()
        pan_x = start_pan_x + (float(event.x) - start_x) * (2.0 / max(width, 120.0))
        pan_y = start_pan_y + (float(event.y) - start_y) * (2.0 / max(height, 120.0))
        self.preview_pan_x = max(-PREVIEW_MAX_PAN, min(PREVIEW_MAX_PAN, pan_x))
        self.preview_pan_y = max(-PREVIEW_MAX_PAN, min(PREVIEW_MAX_PAN, pan_y))
        self._schedule_preview_refresh(delay_ms=25)
        return "break"

    def _on_preview_drag_release(self, _event: tk.Event) -> str:
        self._preview_drag_mode = None
        self._preview_drag_origin = None
        return "break"

    def _on_preview_mousewheel(self, event: tk.Event) -> str:
        if not self._preview_interaction_enabled():
            return "break"
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return "break"
        steps = float(delta) / 120.0
        self.preview_zoom = max(PREVIEW_MIN_ZOOM, min(PREVIEW_MAX_ZOOM, self.preview_zoom * (1.12**steps)))
        self._schedule_preview_refresh(delay_ms=10)
        return "break"

    def _render_preview(self, point_cloud_path: str) -> str:
        render_data = self._get_render_data(point_cloud_path)
        artifacts_dir = self._analysis_artifacts_dir(point_cloud_path)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        result = render_point_cloud_view(
            render_data=render_data,
            output_path=str(artifacts_dir / "point_cloud_preview.png"),
            render_style=self._current_render_style(),
            yaw_deg=float(self.preview_yaw_var.get()),
            pitch_deg=float(self.preview_pitch_var.get()),
            pan_x=self.preview_pan_x,
            pan_y=self.preview_pan_y,
            zoom=self.preview_zoom,
            width=PREVIEW_SIZE,
            height=PREVIEW_SIZE,
            log=self._log,
        )
        return result.image_path

    def _prepare_view_assets(self, point_cloud_path: str) -> tuple[str, list[str], str]:
        render_data = self._get_render_data(point_cloud_path)
        artifacts_dir = self._analysis_artifacts_dir(point_cloud_path)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        preview_path = self._render_preview(point_cloud_path)
        view_set = render_point_cloud_view_set(
            render_data=render_data,
            output_dir=str(artifacts_dir / "api_views"),
            render_style=self._current_render_style(),
            width=API_VIEW_SIZE,
            height=API_VIEW_SIZE,
            log=self._log,
        )
        snapshot = render_point_cloud_snapshot(
            render_data=render_data,
            output_path=str(artifacts_dir / "point_cloud_snapshot.png"),
            render_style=self._current_render_style(),
            log=self._log,
        )
        return preview_path, list(view_set.image_paths), snapshot.image_path

    def _set_busy(self, busy: bool) -> None:
        widgets = (
            self.work_dir_entry,
            self.video_entry,
            self.point_cloud_entry,
            self.colmap_entry,
            self.pipeline_combo,
            self.dim_quality_combo,
            self.render_style_combo,
            self.video_fps_entry,
            self.video_max_frames_entry,
            self.video_blur_entry,
            self.video_dedupe_entry,
            self.video_min_gap_entry,
            self.overwrite_check,
            self.use_dim_env_check,
            self.colmap_auto_btn,
            self.run_btn,
            self.prepare_views_btn,
            self.run_and_analyze_btn,
            self.analyze_current_btn,
            self.open_meshlab_btn,
            self.preview_left_btn,
            self.preview_right_btn,
            self.preview_up_btn,
            self.preview_down_btn,
            self.preview_refresh_btn,
            self.preview_reset_btn,
            self.preview_yaw_scale,
            self.preview_pitch_scale,
        )
        state = "disabled" if busy else "normal"
        readonly_widgets = {self.dim_quality_combo, self.render_style_combo}
        for widget in widgets:
            widget.configure(state="readonly" if not busy and widget in readonly_widgets else state)

    def _with_thread(self, title: str, worker_fn) -> None:
        if self.running:
            return
        self.running = True
        self._set_busy(True)
        self._log(f"===== {title} =====")

        def runner() -> None:
            try:
                worker_fn()
            except Exception as exc:  # noqa: BLE001
                self._log(f"[ERROR] {exc}")
                self.root.after(0, lambda message=f"{title}失败：\n{exc}": self._set_text(self.result_box, message))
                self.root.after(0, lambda: messagebox.showerror("错误", str(exc)))
            finally:
                self.root.after(0, self._on_finished)

        threading.Thread(target=runner, daemon=True).start()

    def run_pipeline_thread(self) -> None:
        def worker() -> None:
            if self.point_cloud_var.get().strip() and not self.video_var.get().strip():
                self.root.after(0, lambda: self._set_point_cloud_path(self.point_cloud_var.get().strip()))
                return
            cfg = self._collect_pipeline_config()
            point_cloud_path = run_pipeline(cfg, log=self._log)
            self.root.after(0, lambda path=point_cloud_path: self._set_point_cloud_path(path))
            preview_path, view_paths, snapshot_path = self._prepare_view_assets(point_cloud_path)
            self.root.after(0, lambda: self._set_preview_image(preview_path))
            self.root.after(0, lambda: self._set_api_view_images(view_paths))
            self.root.after(0, lambda: self.snapshot_path_var.set(snapshot_path))

        self._save_config()
        self._with_thread("生成点云", worker)

    def prepare_views_thread(self) -> None:
        point_cloud_path = self.point_cloud_var.get().strip()
        if not point_cloud_path:
            messagebox.showerror("参数错误", "请先选择点云文件，或先生成点云。")
            return

        def worker() -> None:
            preview_path, view_paths, snapshot_path = self._prepare_view_assets(point_cloud_path)
            self.root.after(0, lambda: self._set_preview_image(preview_path))
            self.root.after(0, lambda: self._set_api_view_images(view_paths))
            self.root.after(0, lambda: self.snapshot_path_var.set(snapshot_path))
            self.root.after(
                0,
                lambda: self._set_text(
                    self.result_box,
                    "四张 API 图片已生成：\n" + "\n".join(f"- {path}" for path in view_paths),
                ),
            )

        self._save_config()
        self._with_thread("生成四图", worker)

    def configure_colmap_thread(self) -> None:
        def worker() -> None:
            result = ensure_colmap_binary(
                preferred_path=self.colmap_var.get().strip(),
                download_if_missing=True,
                log=self._log,
            )
            self.root.after(0, lambda path=result.binary_path: self.colmap_var.set(path))
            self.root.after(0, self._save_config)
            self._log(f"[OK] COLMAP configured ({result.source}): {result.binary_path}")

        self._save_config()
        self._with_thread("自动配置 COLMAP", worker)

    def open_point_cloud_in_meshlab_thread(self) -> None:
        point_cloud_path = self.point_cloud_var.get().strip()
        if not point_cloud_path:
            messagebox.showerror("参数错误", "请先选择点云文件，或先生成点云。")
            return

        def worker() -> None:
            binary_path = self._ensure_meshlab_binary_for_run(download_if_missing=True)
            open_point_cloud_in_meshlab(
                point_cloud_path=point_cloud_path,
                preferred_path=binary_path,
                download_if_missing=False,
                log=self._log,
            )

        self._with_thread("打开 MeshLab", worker)

    def _analyze_point_cloud(self, point_cloud_path: str) -> None:
        preview_path, view_paths, snapshot_path = self._prepare_view_assets(point_cloud_path)
        prompt, api_key, base_url, model, detail = self._collect_api_settings()
        artifacts_dir = self._analysis_artifacts_dir(point_cloud_path)
        result = analyze_images_with_openai(
            image_paths=view_paths,
            prompt=prompt,
            api_key=api_key,
            base_url=base_url,
            model=model,
            detail=detail,
            response_path=str((artifacts_dir / "openai_response.json").resolve()),
            log=self._log,
        )
        self.root.after(0, lambda: self._set_preview_image(preview_path))
        self.root.after(0, lambda: self._set_api_view_images(view_paths))
        self.root.after(0, lambda: self.snapshot_path_var.set(snapshot_path))
        self.root.after(0, lambda: self.response_path_var.set(result.response_path))
        self.root.after(
            0,
            lambda: self._set_text(
                self.result_box,
                "本次上传的四张图片：\n"
                + "\n".join(f"- {path}" for path in view_paths)
                + "\n\nAPI 回答：\n"
                + result.text,
            ),
        )

    def analyze_current_point_cloud_thread(self) -> None:
        point_cloud_path = self.point_cloud_var.get().strip()
        if not point_cloud_path:
            messagebox.showerror("参数错误", "请先选择点云文件，或先生成点云。")
            return

        def worker() -> None:
            self._analyze_point_cloud(point_cloud_path)

        self._save_config()
        self._with_thread("分析当前点云", worker)

    def run_and_analyze_thread(self) -> None:
        def worker() -> None:
            point_cloud_path = self.point_cloud_var.get().strip()
            if not point_cloud_path or self.video_var.get().strip():
                cfg = self._collect_pipeline_config()
                if cfg.video_path:
                    generated = run_pipeline(cfg, log=self._log)
                    self.root.after(0, lambda path=generated: self._set_point_cloud_path(path))
                    point_cloud_path_final = generated
                else:
                    point_cloud_path_final = self.point_cloud_var.get().strip()
            else:
                point_cloud_path_final = point_cloud_path
            self._analyze_point_cloud(point_cloud_path_final)

        self._save_config()
        self._with_thread("生成并分析", worker)

    def refresh_preview_thread(self) -> None:
        self._schedule_preview_refresh()

    def reset_preview_thread(self) -> None:
        self.preview_yaw_var.set(35.0)
        self.preview_pitch_var.set(-25.0)
        self.preview_pan_x = 0.0
        self.preview_pan_y = 0.0
        self.preview_zoom = 1.0
        self._update_preview_status(prefix="已重置视角。")
        self.refresh_preview_thread()

    def _on_finished(self) -> None:
        self.running = False
        self._set_busy(False)
        self._log("===== 结束 =====")

    def _apply_saved_config(self) -> None:
        cfg = self._saved_config or {}

        def _set(var: tk.StringVar, key: str) -> None:
            value = cfg.get(key)
            if value is not None:
                var.set(str(value))

        _set(self.work_dir_var, "work_dir")
        _set(self.video_var, "video_path")
        _set(self.point_cloud_var, "point_cloud_path")
        _set(self.colmap_var, "colmap_bin")
        _set(self.pipeline_var, "pipeline")
        _set(self.dim_quality_var, "dim_quality")
        _set(self.render_style_label_var, "render_style_label")
        _set(self.video_sample_fps_var, "video_sample_fps")
        _set(self.video_max_frames_var, "video_max_frames")
        _set(self.video_blur_threshold_var, "video_blur_threshold")
        _set(self.video_dedupe_threshold_var, "video_dedupe_threshold")
        _set(self.video_min_gap_sec_var, "video_min_gap_sec")
        if OPENAI_BASE_URL_ENV not in self._runtime_openai_defaults:
            _set(self.openai_base_url_var, "openai_base_url")
        if OPENAI_MODEL_ENV not in self._runtime_openai_defaults:
            _set(self.openai_model_var, "openai_model")
        _set(self.openai_detail_var, "openai_detail")
        _set(self.response_path_var, "response_path")
        _set(self.snapshot_path_var, "snapshot_path")
        self.overwrite_var.set(bool(cfg.get("overwrite", True)))
        self.use_dim_env_var.set(bool(cfg.get("use_dim_env", True)))
        self.preview_yaw_var.set(float(cfg.get("preview_yaw", 35.0)))
        self.preview_pitch_var.set(float(cfg.get("preview_pitch", -25.0)))
        self.preview_pan_x = float(cfg.get("preview_pan_x", 0.0))
        self.preview_pan_y = float(cfg.get("preview_pan_y", 0.0))
        self.preview_zoom = float(cfg.get("preview_zoom", 1.0))
        saved_prompt = str(cfg.get("prompt") or "").strip()
        if not saved_prompt or saved_prompt in LEGACY_ANALYSIS_PROMPTS:
            saved_prompt = DEFAULT_ANALYSIS_PROMPT
        self.prompt_box.insert("1.0", saved_prompt)
        current_colmap = self.colmap_var.get().strip()
        if not current_colmap or current_colmap == "colmap" or not Path(current_colmap).exists():
            detected = detect_colmap_binary()
            if detected is not None:
                self.colmap_var.set(detected.binary_path)
        detected_meshlab = detect_meshlab_binary()
        if detected_meshlab is not None:
            self._meshlab_binary_path = detected_meshlab.binary_path
        if self.point_cloud_var.get().strip():
            self._update_preview_status(prefix="已恢复上次视角。")

    def _apply_runtime_constraints(self) -> None:
        if not getattr(sys, "frozen", False):
            return
        self.use_dim_env_var.set(True)
        self.use_dim_env_check.configure(state="disabled")
        self._log("[INFO] Frozen workbench forces the managed DIM env. Source-tree mode is disabled in packaged builds.")

    def _save_config(self) -> None:
        update_section(
            "workbench_gui",
            {
                "work_dir": self.work_dir_var.get().strip(),
                "video_path": self.video_var.get().strip(),
                "point_cloud_path": self.point_cloud_var.get().strip(),
                "colmap_bin": self.colmap_var.get().strip(),
                "pipeline": self.pipeline_var.get().strip(),
                "dim_quality": self.dim_quality_var.get().strip(),
                "render_style_label": self.render_style_label_var.get().strip() or "面状",
                "video_sample_fps": self.video_sample_fps_var.get().strip(),
                "video_max_frames": self.video_max_frames_var.get().strip(),
                "video_blur_threshold": self.video_blur_threshold_var.get().strip(),
                "video_dedupe_threshold": self.video_dedupe_threshold_var.get().strip(),
                "video_min_gap_sec": self.video_min_gap_sec_var.get().strip(),
                "overwrite": self.overwrite_var.get(),
                "use_dim_env": self.use_dim_env_var.get(),
                "openai_base_url": self.openai_base_url_var.get().strip() or DEFAULT_OPENAI_BASE_URL,
                "openai_model": self.openai_model_var.get().strip() or DEFAULT_OPENAI_MODEL,
                "openai_detail": self.openai_detail_var.get().strip() or "low",
                "preview_yaw": f"{self.preview_yaw_var.get():.2f}",
                "preview_pitch": f"{self.preview_pitch_var.get():.2f}",
                "preview_pan_x": f"{self.preview_pan_x:.4f}",
                "preview_pan_y": f"{self.preview_pan_y:.4f}",
                "preview_zoom": f"{self.preview_zoom:.4f}",
                "snapshot_path": self.snapshot_path_var.get().strip(),
                "response_path": self.response_path_var.get().strip(),
                "prompt": self.prompt_box.get("1.0", tk.END).strip() or DEFAULT_ANALYSIS_PROMPT,
                "openai_api_key": "",
            },
        )

    def _on_close(self) -> None:
        if self._preview_refresh_after_id is not None:
            self.root.after_cancel(self._preview_refresh_after_id)
        self._save_config()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    WorkbenchGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
