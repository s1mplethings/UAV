"""
Utility to auto-manage a dedicated Python 3.9 environment for deep-image-matching.

Usage::
    dim_env = DeepImageMatchingEnv()
    dim_env.run_dim(dir=r"D:\UAV\data\my_scene", pipeline="superpoint+lightglue")

When packaged with PyInstaller, the environment will be created next to the
executable on first run (folder name ``py39_dim_env`` by default).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

LogFn = Callable[[str], None]


class DeepImageMatchingEnv:
    """
    Manage a standalone conda env (Python 3.9 + deep-image-matching) and run DIM inside it.

    - Looks for conda in PATH.
    - Creates a local env in ``<base_dir>/<env_name>`` (``py39_dim_env`` by default).
    - Installs deep-image-matching in that env and reuses it across runs.
    """

    def __init__(self, env_name: str = "py39_dim_env", log_fn: LogFn | None = None):
        # Compatible with PyInstaller: sys._MEIPASS points to the unpacked temp dir.
        if hasattr(sys, "_MEIPASS"):
            base_dir = Path(sys._MEIPASS)  # type: ignore[attr-defined]
        else:
            base_dir = Path(__file__).resolve().parent

        self.base_dir = base_dir
        self.env_dir = self.base_dir / env_name
        # conda -p puts python.exe directly under env_dir on Windows; use bin/python on POSIX.
        if os.name == "nt":
            self.python_exe = self.env_dir / "python.exe"
            self._alt_python = self.env_dir / "Scripts" / "python.exe"
        else:
            self.python_exe = self.env_dir / "bin" / "python"
            self._alt_python = self.python_exe

        self.log: LogFn = log_fn or (lambda msg: print(msg))

    def _find_conda(self) -> str:
        conda = shutil.which("conda")
        if not conda:
            raise RuntimeError("未找到 conda，请保证 conda 在 PATH 中。")
        return conda

    def _run(self, cmd: Sequence[str], env: dict[str, str] | None = None, check: bool = True):
        self.log("[DIM ENV] " + " ".join(map(str, cmd)))
        result = subprocess.run(
            cmd,
            env=env,
            check=check,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.stdout:
            # Trim trailing whitespace to keep logs tidy.
            for line in result.stdout.rstrip().splitlines():
                self.log(line)
        return result

    def ensure_env(self) -> Path:
        """
        Make sure the dedicated env exists and has deep-image-matching installed.
        Returns the path to python executable within that env.
        """
        if self.python_exe.exists():
            return self.python_exe

        conda = self._find_conda()
        self.env_dir.mkdir(parents=True, exist_ok=True)

        create_cmd = [
            conda,
            "create",
            "-y",
            "-p",
            str(self.env_dir),
            "python=3.9",
            "-c",
            "https://repo.anaconda.com/pkgs/main",
            "--override-channels",
        ]
        self._run(create_cmd)

        if not self.python_exe.exists():
            # Rare fallback: some setups place python.exe under Scripts on Windows.
            if self._alt_python.exists():
                self.python_exe = self._alt_python
            else:
                raise RuntimeError("创建 py39_dim_env 失败，未找到 python 可执行文件。")

        # Install DIM deps in the new env.
        self._run([str(self.python_exe), "-m", "pip", "install", "--upgrade", "pip"])
        self._run([str(self.python_exe), "-m", "pip", "install", "deep-image-matching"])

        return self.python_exe

    def run_dim(self, dir: str, pipeline: str = "superpoint+lightglue", extra_args: Iterable[str] | None = None):
        """
        Run deep-image-matching inside the managed Python 3.9 environment.
        ``dir`` must contain an ``images`` subdirectory.
        """
        dir_path = Path(dir)
        if not dir_path.exists():
            raise FileNotFoundError(f"指定目录不存在: {dir_path}")

        images_dir = dir_path / "images"
        if not images_dir.exists():
            raise FileNotFoundError(f"{dir_path} 下找不到 images 目录")

        env_python = self.ensure_env()

        cmd = [
            str(env_python),
            "-m",
            "deep_image_matching",
            "--dir",
            str(dir_path),
            "--pipeline",
            pipeline,
        ]

        if extra_args:
            cmd.extend(extra_args)

        self._run(cmd)


def run_dim_for_scene(scene_dir: str, pipeline: str = "superpoint+lightglue", extra_args: Iterable[str] | None = None):
    """
    Convenience wrapper if you don't want to instantiate the class yourself.
    """
    dim_env = DeepImageMatchingEnv()
    dim_env.run_dim(dir=scene_dir, pipeline=pipeline, extra_args=extra_args)


if __name__ == "__main__":
    # Simple manual test: update the path before running this file directly.
    scene = r"D:\UAV\data\my_scene"
    run_dim_for_scene(scene)
