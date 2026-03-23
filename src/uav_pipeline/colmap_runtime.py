from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from .runtime_download import download_file_with_resume, request_json

LogFn = Callable[[str], None]

COLMAP_RELEASE_API_URL = "https://api.github.com/repos/colmap/colmap/releases/latest"
COLMAP_WINDOWS_CUDA_ASSET = "colmap-x64-windows-cuda.zip"
COLMAP_WINDOWS_NO_CUDA_ASSET = "colmap-x64-windows-nocuda.zip"


@dataclass(frozen=True)
class ColmapBinaryResult:
    binary_path: str
    source: str
    install_dir: str | None = None
    release_tag: str | None = None
    asset_name: str | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runtime_tools_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "_internal" / "runtime_tools" / "colmap"
    return _repo_root() / ".runtime_tools" / "colmap"


def _manifest_path() -> Path:
    return _runtime_tools_root() / "installed.json"


def _normalize_colmap_candidate(path: str | os.PathLike[str] | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        exe = candidate / "bin" / "colmap.exe"
        if exe.exists():
            return exe.resolve()
        batch = candidate / "COLMAP.bat"
        if batch.exists():
            exe = batch.parent / "bin" / "colmap.exe"
            if exe.exists():
                return exe.resolve()
            return batch.resolve()
        return None

    if not candidate.exists():
        which = shutil.which(str(candidate))
        if which:
            candidate = Path(which)
        else:
            return None

    if candidate.name.lower() == "colmap.bat":
        exe = candidate.parent / "bin" / "colmap.exe"
        if exe.exists():
            return exe.resolve()
    return candidate.resolve()


def _common_windows_candidates() -> tuple[Path, ...]:
    return (
        Path(r"C:\Program Files\COLMAP\bin\colmap.exe"),
        Path(r"C:\Program Files\COLMAP\COLMAP.bat"),
        Path(r"C:\Program Files (x86)\COLMAP\bin\colmap.exe"),
        Path(r"C:\Program Files (x86)\COLMAP\COLMAP.bat"),
    )


def _runtime_candidates() -> Iterable[Path]:
    runtime_root = _runtime_tools_root()
    manifest = _manifest_path()
    if manifest.exists():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            installed_path = payload.get("binary_path")
            resolved = _normalize_colmap_candidate(installed_path)
            if resolved is not None:
                yield resolved
        except Exception:
            pass

    if runtime_root.exists():
        for candidate in sorted(runtime_root.rglob("bin/colmap.exe")):
            yield candidate.resolve()


def detect_colmap_binary(preferred_path: str | None = None) -> ColmapBinaryResult | None:
    preferred = _normalize_colmap_candidate(preferred_path)
    if preferred is not None:
        return ColmapBinaryResult(binary_path=str(preferred), source="configured", install_dir=str(preferred.parent.parent))

    for candidate in _runtime_candidates():
        normalized = _normalize_colmap_candidate(candidate)
        if normalized is not None:
            return ColmapBinaryResult(binary_path=str(normalized), source="runtime", install_dir=str(normalized.parent.parent))

    env_candidate = _normalize_colmap_candidate(os.environ.get("COLMAP_BIN"))
    if env_candidate is not None:
        return ColmapBinaryResult(binary_path=str(env_candidate), source="env", install_dir=str(env_candidate.parent.parent))

    which_candidate = _normalize_colmap_candidate("colmap")
    if which_candidate is not None:
        return ColmapBinaryResult(binary_path=str(which_candidate), source="path", install_dir=str(which_candidate.parent.parent))

    if os.name == "nt":
        for candidate in _common_windows_candidates():
            normalized = _normalize_colmap_candidate(candidate)
            if normalized is not None:
                return ColmapBinaryResult(
                    binary_path=str(normalized),
                    source="common_install",
                    install_dir=str(normalized.parent.parent),
                )
    return None


def _has_nvidia_gpu() -> bool:
    if shutil.which("nvidia-smi"):
        try:
            proc = subprocess.run(
                ["nvidia-smi", "-L"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
                check=False,
            )
            return proc.returncode == 0 and bool(proc.stdout.strip())
        except Exception:
            return True
    return False


def _choose_windows_asset(release_json: dict[str, object], prefer_cuda: bool) -> tuple[str, str, str]:
    assets = release_json.get("assets")
    if not isinstance(assets, list):
        raise RuntimeError("COLMAP release metadata does not contain downloadable assets.")

    preferred_names = [COLMAP_WINDOWS_CUDA_ASSET, COLMAP_WINDOWS_NO_CUDA_ASSET] if prefer_cuda else [
        COLMAP_WINDOWS_NO_CUDA_ASSET,
        COLMAP_WINDOWS_CUDA_ASSET,
    ]

    for asset_name in preferred_names:
        for asset in assets:
            if not isinstance(asset, dict):
                continue
            if asset.get("name") != asset_name:
                continue
            url = asset.get("browser_download_url")
            if isinstance(url, str) and url.strip():
                tag_name = str(release_json.get("tag_name", "") or "")
                return asset_name, url, tag_name

    raise RuntimeError("Could not find a Windows COLMAP zip asset in the latest official release.")


def _find_extracted_colmap_root(staging_dir: Path) -> Path:
    for candidate in staging_dir.rglob("bin/colmap.exe"):
        return candidate.parent.parent
    raise RuntimeError("Downloaded COLMAP archive does not contain bin/colmap.exe")


def _write_manifest(result: ColmapBinaryResult) -> None:
    manifest = _manifest_path()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "binary_path": result.binary_path,
                "source": result.source,
                "install_dir": result.install_dir,
                "release_tag": result.release_tag,
                "asset_name": result.asset_name,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def download_colmap_for_windows(*, prefer_cuda: bool | None = None, log: LogFn = print) -> ColmapBinaryResult:
    if os.name != "nt":
        raise RuntimeError("Automatic COLMAP download is currently implemented for Windows only.")

    runtime_root = _runtime_tools_root()
    runtime_root.mkdir(parents=True, exist_ok=True)

    resolved_prefer_cuda = _has_nvidia_gpu() if prefer_cuda is None else prefer_cuda
    release_json = request_json(COLMAP_RELEASE_API_URL)
    asset_name, download_url, tag_name = _choose_windows_asset(release_json, prefer_cuda=resolved_prefer_cuda)
    final_root = runtime_root / (tag_name or "latest")
    final_binary = final_root / "bin" / "colmap.exe"
    if final_binary.exists():
        result = ColmapBinaryResult(
            binary_path=str(final_binary.resolve()),
            source="runtime_cached",
            install_dir=str(final_root.resolve()),
            release_tag=tag_name or None,
            asset_name=asset_name,
        )
        _write_manifest(result)
        return result

    log(f"[INFO] Downloading official COLMAP release {tag_name or '(latest)'}: {asset_name}")
    archive_path = runtime_root / asset_name
    staging_dir = runtime_root / "_staging"

    if archive_path.exists():
        archive_path.unlink()
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    download_file_with_resume(download_url, archive_path, label="COLMAP", log=log)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(staging_dir)

    extracted_root = _find_extracted_colmap_root(staging_dir)
    if final_root.exists():
        shutil.rmtree(final_root)
    final_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(extracted_root), str(final_root))

    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    if archive_path.exists():
        archive_path.unlink()

    result = ColmapBinaryResult(
        binary_path=str((final_root / "bin" / "colmap.exe").resolve()),
        source="runtime_downloaded",
        install_dir=str(final_root.resolve()),
        release_tag=tag_name or None,
        asset_name=asset_name,
    )
    _write_manifest(result)
    return result


def ensure_colmap_binary(
    *,
    preferred_path: str | None = None,
    download_if_missing: bool = False,
    prefer_cuda: bool | None = None,
    log: LogFn = print,
) -> ColmapBinaryResult:
    detected = detect_colmap_binary(preferred_path=preferred_path)
    if detected is not None:
        return detected
    if download_if_missing:
        return download_colmap_for_windows(prefer_cuda=prefer_cuda, log=log)
    raise FileNotFoundError(
        "找不到 COLMAP。请在 GUI 中选择 colmap.exe，或允许 GUI 自动下载官方 Windows 版 COLMAP。"
    )
