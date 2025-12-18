r"""
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

    def __init__(
        self,
        env_name: str = "py39_dim_env",
        log_fn: LogFn | None = None,
        torch_version: str = "2.2.1",
        torchvision_version: str = "0.17.1",
        torch_cuda: str = "cu121",
        torch_index_url: str = "https://download.pytorch.org/whl/cu121",
        install_cuda_torch: bool = True,
        numpy_spec: str = "numpy<2",
    ):
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
        self.torch_version = torch_version
        self.torchvision_version = torchvision_version
        self.torch_cuda = torch_cuda
        self.torch_index_url = torch_index_url
        self.install_cuda_torch = install_cuda_torch
        self.numpy_spec = numpy_spec

    def _find_conda(self) -> str:
        conda = shutil.which("conda")
        if not conda:
            raise RuntimeError("未找到 conda，请保证 conda 在 PATH 中。")
        return conda

    def _run(self, cmd: Sequence[str], env: dict[str, str] | None = None, check: bool = True):
        """
        Run a command and log stdout even if it fails, so GUI/CLI users can see errors.
        """
        self.log("[DIM ENV] " + " ".join(map(str, cmd)))
        try:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as e:
            exe = str(cmd[0]) if cmd else "<empty>"
            raise FileNotFoundError(
                f"系统找不到可执行文件: {exe}\n"
                "如果这是 COLMAP，请在 GUI/CLI 里设置 --colmap_bin 为 colmap.exe 的完整路径，"
                "或把 COLMAP 的 bin 目录加入 PATH。"
            ) from e
        assert proc.stdout is not None
        for line in proc.stdout:
            self.log(line.rstrip())
        proc.wait()
        if check and proc.returncode:
            raise subprocess.CalledProcessError(proc.returncode, cmd)
        return proc.returncode

    def ensure_env(self) -> Path:
        """
        Make sure the dedicated env exists and has deep-image-matching installed.
        Returns the path to python executable within that env.
        """
        # 1) Resolve python executable for an existing env (if any).
        env_python: Path | None = None
        if self.python_exe.exists():
            env_python = self.python_exe
        elif self._alt_python.exists():
            env_python = self._alt_python

        # 2) If env exists and runtime is healthy, we're done.
        if env_python is not None and self._runtime_ok(env_python):
            self.python_exe = env_python
            return self.python_exe

        # 3) If env is missing, create it.
        if env_python is None:
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

            if self.python_exe.exists():
                env_python = self.python_exe
            elif self._alt_python.exists():
                env_python = self._alt_python
                self.python_exe = env_python
            else:
                raise RuntimeError("创建 py39_dim_env 失败，未找到 python 可执行文件。")
        else:
            # Env exists but DIM is missing/broken → repair by (re)installing packages.
            self.python_exe = env_python

        # 4) Install/repair DIM deps in the env.
        self._run([str(self.python_exe), "-m", "pip", "install", "--upgrade", "pip"])

        # Numpy 2.x is not compatible with some PyTorch wheels (especially older versions).
        # Pin numpy to <2 to avoid runtime warnings/crashes when importing torch.
        if self.numpy_spec:
            self._run([str(self.python_exe), "-m", "pip", "install", "--upgrade", "--force-reinstall", self.numpy_spec])

        # Prefer a CUDA build of torch/torchvision if requested.
        if self.install_cuda_torch:
            torch_pkgs = []
            if self.torch_version:
                torch_pkgs.append(f"torch=={self.torch_version}+{self.torch_cuda}")
            if self.torchvision_version:
                torch_pkgs.append(f"torchvision=={self.torchvision_version}+{self.torch_cuda}")
            if torch_pkgs:
                self._run(
                    [
                        str(self.python_exe),
                        "-m",
                        "pip",
                        "install",
                        "--upgrade",
                        "--force-reinstall",
                        *torch_pkgs,
                        "--index-url",
                        self.torch_index_url,
                    ]
                )

        self._run([str(self.python_exe), "-m", "pip", "install", "deep-image-matching"])

        # Some dependencies may pull in NumPy 2.x; enforce our pin again at the end.
        if self.numpy_spec:
            self._run([str(self.python_exe), "-m", "pip", "install", "--upgrade", "--force-reinstall", self.numpy_spec])

        # Final sanity check to catch "Numpy is not available" at runtime.
        if not self._runtime_ok(self.python_exe):
            raise RuntimeError(
                "DIM 环境创建完成，但 numpy/torch 在运行时不可用（常见原因：numpy 2.x 与 torch 不兼容）。\n"
                f"请在该环境里执行：{self.python_exe} -m pip install --upgrade --force-reinstall \"{self.numpy_spec}\""
            )

        return self.python_exe

    def _runtime_ok(self, python_exe: Path) -> bool:
        try:
            # Validate numpy + torch interoperability (torch.from_numpy requires numpy C-API).
            script = (
                "import numpy as np\n"
                "import torch\n"
                "import deep_image_matching\n"
                "torch.from_numpy(np.zeros((1,), dtype=np.float32))\n"
                "print('OK', np.__version__, torch.__version__)\n"
            )
            self._run([str(python_exe), "-c", script], check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def _build_env_vars(self, gpu: int | None = None) -> dict[str, str]:
        env_vars = os.environ.copy()
        if gpu is not None:
            env_vars["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # Make sure our source tree is importable inside the managed env.
        env_vars["PYTHONPATH"] = str(self.base_dir.parent)
        return env_vars

    def run_dim_wrapper(self, argv: Sequence[str], gpu: int | None = None) -> None:
        env_python = self.ensure_env()
        cmd = [str(env_python), "-m", "uav_pipeline.dim_wrapper", *map(str, argv)]
        self._run(cmd, env=self._build_env_vars(gpu))

    def list_pipelines(self) -> None:
        """List available DIM pipelines in the managed env."""
        self.run_dim_wrapper(["--list_pipelines"])

    def probe_pipelines(
        self,
        *,
        scene_dir: str,
        pipelines: str = "all",
        quality: str = "lowest",
        gpu: int | None = None,
    ) -> None:
        """Try to initialize pipelines and report OK/FAIL (no matching is run)."""
        self.run_dim_wrapper(
            [
                "--dir",
                scene_dir,
                "--pipelines",
                pipelines,
                "--quality",
                quality,
                "--probe_pipelines",
                "--print_summary",
            ],
            gpu=gpu,
        )

    def test_pipelines(
        self,
        *,
        scene_dir: str,
        pipelines: str = "all",
        output_dir: str | None = None,
        max_images: int | None = None,
        quality: str = "lowest",
        overwrite: bool = False,
        single_camera: bool = True,
        camera_model: str = "simple-radial",
        gpu: int | None = None,
    ) -> None:
        """Run matching for multiple pipelines (optionally limit to first N images)."""
        out_dir = output_dir or str(Path(scene_dir) / "dim_tests")
        argv: list[str] = [
            "--dir",
            scene_dir,
            "--pipelines",
            pipelines,
            "--quality",
            quality,
            "--output",
            out_dir,
            "--camera_model",
            camera_model,
            "--print_summary",
        ]
        if max_images is not None:
            argv += ["--max_images", str(max_images)]
        if overwrite:
            argv.append("--overwrite")
        if not single_camera:
            argv.append("--multi_camera")
        self.run_dim_wrapper(argv, gpu=gpu)

    def run_dim(
        self,
        dir: str,
        pipeline: str = "superpoint+lightglue",
        colmap_bin: str | None = None,
        gpu: int | None = None,
        overwrite: bool = False,
        quality: str = "medium",
        single_camera: bool = True,
        camera_model: str = "simple-radial",
        geom_verification: bool = True,
    ):
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
        # Validate that the images folder contains actual images (not only subfolders).
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
        img_files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
        if not img_files:
            sub_with_imgs: list[Path] = []
            for p in images_dir.iterdir():
                if p.is_dir():
                    if any(q.is_file() and q.suffix.lower() in exts for q in p.iterdir()):
                        sub_with_imgs.append(p)
            hint = ""
            if sub_with_imgs:
                hint = f"（发现子目录含图片：{sub_with_imgs[0].name}，请把图片移动到 images/ 下，或把该子目录内容链接/复制到 images/）"
            raise FileNotFoundError(f"{images_dir} 里没有检测到图片文件{hint}")

        dim_output = dir_path / "dim_outputs"

        argv: list[str] = [
            "--dir",
            str(dir_path),
            "--pipeline",
            pipeline,
            "--output",
            str(dim_output),
        ]
        if overwrite:
            argv.append("--overwrite")
        if quality:
            argv += ["--quality", quality]
        if not single_camera:
            argv.append("--multi_camera")
        if camera_model:
            argv += ["--camera_model", camera_model]

        self.run_dim_wrapper(argv, gpu=gpu)

        # Convert matches to a sparse model via COLMAP mapper (uses host COLMAP binary).
        if colmap_bin:
            db_path = dim_output / "database.db"
            if not db_path.exists():
                raise FileNotFoundError(f"找不到 COLMAP 数据库: {db_path}")

            sparse_dir = dir_path / "sparse"
            if overwrite and sparse_dir.exists():
                shutil.rmtree(sparse_dir)
            sparse_dir.mkdir(parents=True, exist_ok=True)

            if geom_verification:
                geom_cmd = [
                    colmap_bin,
                    "geometric_verification",
                    "--database_path",
                    str(db_path),
                ]
                self._run(geom_cmd)

            mapper_cmd = [
                colmap_bin,
                "mapper",
                "--database_path",
                str(db_path),
                "--image_path",
                str(images_dir),
                "--output_path",
                str(sparse_dir),
            ]
            self._run(mapper_cmd)


def run_dim_for_scene(
    scene_dir: str,
    pipeline: str = "superpoint+lightglue",
    colmap_bin: str | None = None,
    gpu: int | None = None,
    overwrite: bool = False,
    quality: str = "medium",
    single_camera: bool = True,
    camera_model: str = "simple-radial",
    geom_verification: bool = True,
):
    """
    Convenience wrapper if you don't want to instantiate the class yourself.
    """
    dim_env = DeepImageMatchingEnv()
    dim_env.run_dim(
        dir=scene_dir,
        pipeline=pipeline,
        colmap_bin=colmap_bin,
        gpu=gpu,
        overwrite=overwrite,
        quality=quality,
        single_camera=single_camera,
        camera_model=camera_model,
        geom_verification=geom_verification,
    )


if __name__ == "__main__":
    # Simple manual test: update the path before running this file directly.
    scene = r"D:\UAV\data\my_scene"
    run_dim_for_scene(scene)
