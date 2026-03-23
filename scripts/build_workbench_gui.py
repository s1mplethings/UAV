from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

EXCLUDED_MODULES = (
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
)


def _prepare_runtime_support_package(repo_root: Path, build_root: Path) -> Path:
    """
    Build a lightweight source package that the managed DIM env can import at runtime.

    The frozen GUI runs fine from the PyInstaller archive, but the separate Python 3.9
    env created next to the executable needs real `.py` files on `PYTHONPATH` in order
    to execute `python -m uav_pipeline.dim_wrapper`.
    """
    src_package_dir = repo_root / "src" / "uav_pipeline"
    runtime_root = build_root / "runtime_support"
    runtime_package_dir = runtime_root / "uav_pipeline"

    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_package_dir.mkdir(parents=True, exist_ok=True)

    for source_file in src_package_dir.glob("*.py"):
        shutil.copy2(source_file, runtime_package_dir / source_file.name)

    return runtime_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the standalone uav-workbench-gui Windows package with PyInstaller."
    )
    parser.add_argument(
        "--name",
        default="uav-workbench-gui",
        help="PyInstaller app name. Default: %(default)s",
    )
    parser.add_argument(
        "--mode",
        choices=("onedir", "onefile"),
        default="onedir",
        help="Packaging mode. Default: %(default)s",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="Keep a console window for debugging instead of using windowed mode.",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Skip the final zip archive and only keep the PyInstaller dist folder.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    entry_script = repo_root / "scripts" / "run_workbench_gui.py"
    dist_dir = repo_root / "dist"
    build_root = repo_root / "build" / "workbench_gui"
    spec_dir = build_root / "spec"
    work_dir = build_root / "work"
    runtime_support_root = _prepare_runtime_support_package(repo_root, build_root)

    if not entry_script.exists():
        raise FileNotFoundError(f"Workbench entry script not found: {entry_script}")

    pyinstaller_cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--name",
        args.name,
        "--paths",
        str(repo_root / "src"),
        "--add-data",
        f"{runtime_support_root};runtime_support",
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir),
        "--specpath",
        str(spec_dir),
    ]
    for module_name in EXCLUDED_MODULES:
        pyinstaller_cmd += ["--exclude-module", module_name]
    pyinstaller_cmd.append("--windowed" if not args.console else "--console")
    pyinstaller_cmd.append(f"--{args.mode}")
    pyinstaller_cmd.append(str(entry_script))

    print("[BUILD]", " ".join(pyinstaller_cmd))
    subprocess.run(pyinstaller_cmd, cwd=repo_root, check=True)

    package_dir = dist_dir / args.name
    if args.mode == "onefile":
        package_target = package_dir.with_suffix(".exe")
    else:
        package_target = package_dir

    if not package_target.exists():
        raise FileNotFoundError(f"Expected packaged target was not created: {package_target}")

    print(f"[OK] Packaged target: {package_target}")

    if args.no_zip:
        return 0

    archive_base = dist_dir / f"{args.name}-windows"
    if package_target.is_dir():
        archive_path = Path(shutil.make_archive(str(archive_base), "zip", dist_dir, args.name))
    else:
        archive_path = archive_base.with_suffix(".zip")
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(package_target, arcname=package_target.name)
    print(f"[OK] Zip archive: {archive_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
