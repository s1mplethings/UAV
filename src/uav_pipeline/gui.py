"""Tkinter GUI for the deep-image-matching + COLMAP pipeline."""

from __future__ import annotations

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

from .pipeline import PipelineConfig, run_pipeline


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

        # Flags
        flags = tk.Frame(frame)
        flags.grid(row=6, column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.use_dim_env_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            flags,
            text="Use managed Py3.9 DIM env (conda)",
            variable=self.use_dim_env_var,
        ).pack(side=tk.LEFT, padx=(0, 12))
        self.skip_dim_var = tk.BooleanVar(value=False)
        self.overwrite_var = tk.BooleanVar(value=False)
        tk.Checkbutton(flags, text="跳过 deep-image-matching (--skip_dim)", variable=self.skip_dim_var).pack(
            side=tk.LEFT, padx=(0, 12)
        )
        tk.Checkbutton(flags, text="DIM 覆盖已有输出 (--overwrite)", variable=self.overwrite_var).pack(side=tk.LEFT)

        # Run button
        btn = tk.Button(frame, text="开始运行", command=self.run_pipeline_thread, width=20)
        btn.grid(row=7, column=0, columnspan=3, pady=(12, 0))
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
            )
        except ValueError:
            messagebox.showerror("参数错误", "GPU 参数必须是整数或留空。")
            return

        self.running = True
        self.run_btn.configure(state="disabled")
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

    def _on_finished(self) -> None:
        self.running = False
        self.run_btn.configure(state="normal")
        self._log("===== 结束 =====")


def main() -> None:
    root = tk.Tk()
    PipelineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
