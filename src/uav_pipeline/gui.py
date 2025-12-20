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
import tkinter as tk
import sys
import shutil
from pathlib import Path
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk

from .dim_env import DeepImageMatchingEnv
from .pipeline import PipelineConfig, run_cmd, run_colmap_mvs, run_dim as run_dim_step, run_pipeline

DIM_QUALITY_OPTIONS = ("highest", "high", "medium", "low", "lowest")
CAMERA_MODEL_OPTIONS = ("simple-radial", "simple-pinhole", "pinhole", "opencv")

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
        root.minsize(920, 720)

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.running = False

        self._style()
        self._build_layout()
        self._build_log()
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
        self.notebook.add(self.run_tab, text="运行")
        self.notebook.add(self.test_tab, text="测试")

        self._build_run_tab(self.run_tab.content)
        self._build_test_tab(self.test_tab.content)

    def _build_run_tab(self, parent: ttk.Frame) -> None:
        # Vars
        self.work_dir_var = tk.StringVar()
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
        self._grid_labeled_entry(project, "工作目录 (必须含 images/):", self.work_dir_var, row=0, browse="dir")
        self._grid_labeled_entry(project, "COLMAP 可执行文件:", self.colmap_var, row=1, browse="file")
        self._grid_labeled_entry(project, "Dense 输出目录(可空):", self.dense_dir_var, row=2, browse="dir")

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
        self.run_btn = ttk.Button(actions, text="开始运行", command=self.run_pipeline_thread)
        self.run_btn.pack(side=tk.LEFT)
        ttk.Label(
            actions,
            text="提示：第一次使用 managed env 会安装依赖/下载权重，耗时较长。",
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

    def _grid_labeled_entry(
        self,
        parent: ttk.Frame,
        label: str,
        var: tk.StringVar,
        row: int,
        *,
        browse: str | None = None,
    ) -> None:
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

    def _str_to_int(self, value: str) -> int | None:
        value = value.strip()
        return int(value) if value else None

    def _set_busy(self, busy: bool) -> None:
        if busy:
            for w in (
                self.run_btn,
                self.preset_pipelines_btn,
                self.list_pipelines_btn,
                self.probe_pipelines_btn,
                self.test_pipelines_btn,
                self.mode_combo,
                self.pipeline_combo,
                self.dim_quality_combo,
                self.dim_camera_model_combo,
                self.dim_gpu_entry,
                self.pm_gpu_entry,
                self.use_dim_env_check,
                self.overwrite_check,
                self.skip_dim_check,
                self.multi_cam_check,
                self.skip_geom_check,
            ):
                w.configure(state="disabled")
            return

        # Restore interactive state.
        self.run_btn.configure(state="normal")
        self.preset_pipelines_btn.configure(state="normal")
        self.list_pipelines_btn.configure(state="normal")
        self.probe_pipelines_btn.configure(state="normal")
        self.test_pipelines_btn.configure(state="normal")

        self.mode_combo.configure(state="readonly")
        self.pipeline_combo.configure(state="normal")
        self.dim_quality_combo.configure(state="readonly")
        self.dim_camera_model_combo.configure(state="readonly")
        self.dim_gpu_entry.configure(state="normal")
        self.pm_gpu_entry.configure(state="normal")
        self.use_dim_env_check.configure(state="normal")
        self.overwrite_check.configure(state="normal")
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
        elif mode == MODE_DIM_ONLY:
            self.skip_dim_var.set(False)
            self.skip_dim_check.configure(state="disabled")
            self.pipeline_combo.configure(state="normal")
            self.dim_quality_combo.configure(state="readonly")
            self.dim_camera_model_combo.configure(state="readonly")
            self.multi_cam_check.configure(state="normal")
            self.skip_geom_check.configure(state="normal")
            self.pm_gpu_entry.configure(state="disabled")
        else:
            self.skip_dim_check.configure(state="normal")
            self.pipeline_combo.configure(state="normal")
            self.dim_quality_combo.configure(state="readonly")
            self.dim_camera_model_combo.configure(state="readonly")
            self.multi_cam_check.configure(state="normal")
            self.skip_geom_check.configure(state="normal")
            self.pm_gpu_entry.configure(state="normal")

    def _show_help_for(self, label: str) -> None:
        txt = {
            "模式:": (
                f"{MODE_FULL}: 先跑 DIM 导出 + COLMAP mapper 得到 sparse，再跑 dense。\n"
                f"{MODE_DIM_ONLY}: 只生成 sparse（适合先看能否稀疏重建）。\n"
                f"{MODE_DENSE_ONLY}: 只跑 dense（需要 work_dir 下已有 sparse/）。"
            ),
            "DIM pipeline:": "选择匹配模型组合；常用推荐：aliked+lightglue / sift+lightglue / loftr。",
            "DIM quality:": "分辨率预设；high/highest 更稳但更慢，lowest 用于快速测试。",
            "DIM camera model:": "写入 COLMAP 数据库的相机模型；单相机 UAV 通常 simple-radial 即可。",
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

    def run_pipeline_thread(self) -> None:
        if self.running:
            return

        work_dir = self.work_dir_var.get().strip()
        if not work_dir:
            messagebox.showerror("参数错误", "请先选择工作目录（包含 images/）。")
            return
        try:
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
            )
        except ValueError:
            messagebox.showerror("参数错误", "GPU 参数必须是整数或留空。")
            return

        self.running = True
        self._set_busy(True)
        mode = self.mode_var.get().strip()
        self._log(f"===== 开始运行 ({mode}) =====")

        def worker() -> None:
            try:
                if mode == MODE_DIM_ONLY:
                    run_dim_step(cfg, log=self._log)
                    self._log("[DONE] DIM + sparse 已完成（查看 work_dir 下 sparse/ 与 dim_outputs/）")
                elif mode == MODE_DENSE_ONLY:
                    fused = run_colmap_mvs(cfg, log=self._log)
                    self._log(f"[DONE] Dense 点云输出: {fused}")
                else:
                    fused = run_pipeline(cfg, log=self._log)
                    self._log(f"[DONE] Dense 点云输出: {fused}")
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
            messagebox.showerror("参数错误", "请先选择工作目录（包含 images/）。")
            return

        try:
            pipelines, _max_images, quality = self._get_test_params()
        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            return

        self.running = True
        self._set_busy(True)
        self._log("===== Probe DIM pipelines（仅初始化，不跑匹配） =====")

        def worker() -> None:
            try:
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
            messagebox.showerror("参数错误", "请先选择工作目录（包含 images/）。")
            return

        try:
            pipelines, max_images, quality = self._get_test_params()
            do_bench, bench_interval = self._get_benchmark_params()
        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            return

        self.running = True
        self._set_busy(True)
        run_dense = self.test_run_dense_var.get()
        self._log("===== 跑 DIM pipelines 测试 =====" + (" + Dense" if run_dense else ""))

        def worker() -> None:
            try:
                colmap_bin = self.colmap_var.get().strip() or "colmap"
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


def main() -> None:
    root = tk.Tk()
    PipelineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
