"""Tkinter GUI for the deep-image-matching + COLMAP pipeline."""

from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
import sys
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

from .dim_env import DeepImageMatchingEnv
from .pipeline import PipelineConfig, run_cmd, run_pipeline


class PipelineGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("UAV DIM + COLMAP Pipeline")
        root.geometry("720x640")

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.running = False

        self._build_form()
        self._build_log()
        self._poll_logs()

    def _build_form(self) -> None:
        frame = tk.Frame(self.root)
        frame.pack(fill=tk.X, padx=12, pady=12)

        # Work dir
        self.work_dir_var = tk.StringVar()
        self._add_labeled_entry(frame, "工作目录 (--dir):", self.work_dir_var, row=0, browse="dir")

        # Colmap bin
        self.colmap_var = tk.StringVar(value="colmap")
        self._add_labeled_entry(
            frame, "COLMAP 可执行文件 (--colmap_bin):", self.colmap_var, row=1, browse="file"
        )

        # Pipeline name
        self.pipeline_var = tk.StringVar(value="superpoint+lightglue")
        self._add_labeled_entry(frame, "deep-image-matching pipeline (--pipeline):", self.pipeline_var, row=2)

        # Dense dir (optional)
        self.dense_dir_var = tk.StringVar()
        self._add_labeled_entry(frame, "Dense 输出目录 (--dense_dir，可空):", self.dense_dir_var, row=3, browse="dir")

        # GPU options
        self.gpu_var = tk.StringVar()
        self._add_labeled_entry(frame, "DIM GPU (--gpu，可空):", self.gpu_var, row=4)
        self.pm_gpu_var = tk.StringVar()
        self._add_labeled_entry(frame, "PatchMatch GPU (--patch_match_gpu，可空):", self.pm_gpu_var, row=5)

        # DIM 分辨率预设
        self.dim_quality_var = tk.StringVar(value="medium")
        self._add_labeled_entry(frame, "DIM 分辨率预设 (--dim_quality):", self.dim_quality_var, row=6)

        # DIM 相机模型（写入 database.db 时使用）
        self.dim_camera_model_var = tk.StringVar(value="simple-radial")
        self._add_labeled_entry(frame, "DIM 相机模型 (--dim_camera_model):", self.dim_camera_model_var, row=7)

        # DIM pipelines 测试
        self.test_pipelines_var = tk.StringVar(value="all")
        self._add_labeled_entry(frame, "DIM pipelines 测试列表 (all/逗号分隔):", self.test_pipelines_var, row=8)
        self.test_max_images_var = tk.StringVar(value="2")
        self._add_labeled_entry(frame, "smoke test 使用前 N 张图 (--test_max_images):", self.test_max_images_var, row=9)
        self.test_quality_var = tk.StringVar(value="lowest")
        self._add_labeled_entry(frame, "smoke test 分辨率预设 (--test_quality):", self.test_quality_var, row=10)

        test_btns = tk.Frame(frame)
        test_btns.grid(row=11, column=0, columnspan=3, sticky="w", pady=(6, 0))
        self.preset_pipelines_btn = tk.Button(
            test_btns,
            text="填入推荐列表",
            command=self.fill_recommended_pipelines,
        )
        self.preset_pipelines_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.list_pipelines_btn = tk.Button(test_btns, text="列出 DIM pipelines", command=self.list_pipelines_thread)
        self.list_pipelines_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.probe_pipelines_btn = tk.Button(test_btns, text="Probe pipelines", command=self.probe_pipelines_thread)
        self.probe_pipelines_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.test_pipelines_btn = tk.Button(test_btns, text="跑 smoke test", command=self.test_pipelines_thread)
        self.test_pipelines_btn.pack(side=tk.LEFT)

        # Flags
        flags = tk.Frame(frame)
        flags.grid(row=12, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.use_dim_env_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            flags,
            text="Use managed Py3.9 DIM env (conda)",
            variable=self.use_dim_env_var,
        ).pack(side=tk.LEFT, padx=(0, 12))
        self.skip_dim_var = tk.BooleanVar(value=False)
        self.overwrite_var = tk.BooleanVar(value=False)
        self.dim_multi_camera_var = tk.BooleanVar(value=False)
        self.skip_geom_verify_var = tk.BooleanVar(value=False)
        tk.Checkbutton(flags, text="跳过 deep-image-matching (--skip_dim)", variable=self.skip_dim_var).pack(
            side=tk.LEFT, padx=(0, 12)
        )
        tk.Checkbutton(flags, text="DIM 覆盖已有输出 (--overwrite)", variable=self.overwrite_var).pack(side=tk.LEFT)
        tk.Checkbutton(flags, text="DIM 多相机 (--dim_multi_camera)", variable=self.dim_multi_camera_var).pack(
            side=tk.LEFT, padx=(12, 0)
        )
        tk.Checkbutton(flags, text="跳过 geom verify", variable=self.skip_geom_verify_var).pack(side=tk.LEFT, padx=(12, 0))

        # Run button
        btn = tk.Button(frame, text="开始运行", command=self.run_pipeline_thread, width=20)
        btn.grid(row=13, column=0, columnspan=3, pady=(12, 0))
        self.run_btn = btn

    def _add_labeled_entry(
        self, parent: tk.Frame, label: str, var: tk.StringVar, row: int, browse: str | None = None
    ) -> None:
        tk.Label(parent, text=label, anchor="w").grid(row=row, column=0, sticky="w", pady=2)
        entry = tk.Entry(parent, textvariable=var, width=60)
        entry.grid(row=row, column=1, sticky="we", pady=2)
        parent.grid_columnconfigure(1, weight=1)
        if browse:
            if browse == "dir":
                cmd = lambda: self._choose_dir(var)
                text = "浏览..."
            else:
                cmd = lambda: self._choose_file(var)
                text = "选择..."
            tk.Button(parent, text=text, command=cmd).grid(row=row, column=2, padx=4, pady=2)

    def _build_log(self) -> None:
        tk.Label(self.root, text="日志输出：").pack(anchor="w", padx=12, pady=(4, 0))
        log_box = ScrolledText(self.root, height=22, wrap="word", state="disabled")
        log_box.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        self.log_box = log_box

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
        state = "disabled" if busy else "normal"
        self.run_btn.configure(state=state)
        self.preset_pipelines_btn.configure(state=state)
        self.list_pipelines_btn.configure(state=state)
        self.probe_pipelines_btn.configure(state=state)
        self.test_pipelines_btn.configure(state=state)

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
        try:
            cfg = PipelineConfig(
                work_dir=self.work_dir_var.get(),
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
        self._log("===== 开始运行 =====")

        def worker() -> None:
            try:
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

    def _get_test_params(self) -> tuple[str, int, str]:
        pipelines = self.test_pipelines_var.get().strip() or "all"
        quality = self.test_quality_var.get().strip() or "lowest"
        try:
            max_images = int((self.test_max_images_var.get() or "").strip() or "2")
        except ValueError as e:
            raise ValueError("smoke test 的 N 必须是整数") from e
        return pipelines, max_images, quality

    def fill_recommended_pipelines(self) -> None:
        """
        Fill in a curated list of pipelines that are commonly useful for UAV SfM.
        You can still edit the text field afterwards.
        """
        self.test_pipelines_var.set(
            "sift+kornia_matcher,sift+lightglue,aliked+lightglue,superpoint+lightglue,loftr,se2loftr,roma"
        )

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
        except ValueError as e:
            messagebox.showerror("参数错误", str(e))
            return

        self.running = True
        self._set_busy(True)
        self._log("===== 跑 DIM pipelines smoke test =====")

        def worker() -> None:
            try:
                if self.use_dim_env_var.get():
                    self._get_dim_env().test_pipelines(
                        scene_dir=work_dir,
                        pipelines=pipelines,
                        output_dir=None,
                        max_images=max_images,
                        quality=quality,
                        overwrite=self.overwrite_var.get(),
                        single_camera=not self.dim_multi_camera_var.get(),
                        camera_model=self.dim_camera_model_var.get().strip() or "simple-radial",
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
                        "--max_images",
                        str(max_images),
                        "--camera_model",
                        self.dim_camera_model_var.get().strip() or "simple-radial",
                        "--print_summary",
                    ]
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
