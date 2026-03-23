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

MESHLAB_RELEASE_API_URL = "https://api.github.com/repos/cnr-isti-vclab/meshlab/releases/latest"


@dataclass(frozen=True)
class MeshLabBinaryResult:
    binary_path: str
    source: str
    install_dir: str | None = None
    release_tag: str | None = None
    asset_name: str | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runtime_tools_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "_internal" / "runtime_tools" / "meshlab"
    return _repo_root() / ".runtime_tools" / "meshlab"


def _manifest_path() -> Path:
    return _runtime_tools_root() / "installed.json"


def _normalize_meshlab_candidate(path: str | os.PathLike[str] | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        exe = candidate / "meshlab.exe"
        if exe.exists():
            return exe.resolve()
        return None

    if not candidate.exists():
        which = shutil.which(str(candidate))
        if which:
            candidate = Path(which)
        else:
            return None
    return candidate.resolve()


def _common_windows_candidates() -> tuple[Path, ...]:
    return (
        Path(r"C:\Program Files\VCG\MeshLab\meshlab.exe"),
        Path(r"C:\Program Files\MeshLab\meshlab.exe"),
        Path(r"C:\Program Files (x86)\VCG\MeshLab\meshlab.exe"),
        Path(r"C:\Program Files (x86)\MeshLab\meshlab.exe"),
    )


def _runtime_candidates() -> Iterable[Path]:
    runtime_root = _runtime_tools_root()
    manifest = _manifest_path()
    if manifest.exists():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            installed_path = payload.get("binary_path")
            resolved = _normalize_meshlab_candidate(installed_path)
            if resolved is not None:
                yield resolved
        except Exception:
            pass

    if runtime_root.exists():
        for candidate in sorted(runtime_root.rglob("meshlab.exe")):
            yield candidate.resolve()


def detect_meshlab_binary(preferred_path: str | None = None) -> MeshLabBinaryResult | None:
    preferred = _normalize_meshlab_candidate(preferred_path)
    if preferred is not None:
        return MeshLabBinaryResult(binary_path=str(preferred), source="configured", install_dir=str(preferred.parent))

    for candidate in _runtime_candidates():
        normalized = _normalize_meshlab_candidate(candidate)
        if normalized is not None:
            return MeshLabBinaryResult(binary_path=str(normalized), source="runtime", install_dir=str(normalized.parent))

    env_candidate = _normalize_meshlab_candidate(os.environ.get("MESHLAB_BIN"))
    if env_candidate is not None:
        return MeshLabBinaryResult(binary_path=str(env_candidate), source="env", install_dir=str(env_candidate.parent))

    which_candidate = _normalize_meshlab_candidate("meshlab")
    if which_candidate is not None:
        return MeshLabBinaryResult(binary_path=str(which_candidate), source="path", install_dir=str(which_candidate.parent))

    if os.name == "nt":
        for candidate in _common_windows_candidates():
            normalized = _normalize_meshlab_candidate(candidate)
            if normalized is not None:
                return MeshLabBinaryResult(
                    binary_path=str(normalized),
                    source="common_install",
                    install_dir=str(normalized.parent),
                )
    return None


def _choose_windows_asset(release_json: dict[str, object]) -> tuple[str, str, str]:
    assets = release_json.get("assets")
    if not isinstance(assets, list):
        raise RuntimeError("MeshLab release metadata does not contain downloadable assets.")

    def _score_asset(name: str) -> tuple[int, int, int]:
        lowered = name.lower()
        return (
            1 if lowered.endswith(".zip") else 0,
            1 if "windows" in lowered else 0,
            1 if any(token in lowered for token in ("x86_64", "win64", "amd64")) else 0,
        )

    ranked_assets: list[tuple[tuple[int, int, int], dict[str, object]]] = []
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name", "") or "")
        score = _score_asset(name)
        if score[0] and score[1]:
            ranked_assets.append((score, asset))

    if not ranked_assets:
        raise RuntimeError("Could not find a Windows MeshLab zip asset in the latest official release.")

    ranked_assets.sort(key=lambda item: item[0], reverse=True)
    asset = ranked_assets[0][1]
    asset_name = str(asset.get("name", "") or "")
    download_url = str(asset.get("browser_download_url", "") or "")
    if not asset_name or not download_url:
        raise RuntimeError("MeshLab release asset metadata is missing a usable download URL.")
    tag_name = str(release_json.get("tag_name", "") or "")
    return asset_name, download_url, tag_name


def _find_extracted_meshlab_dir(staging_dir: Path) -> Path:
    for candidate in staging_dir.rglob("meshlab.exe"):
        return candidate.parent
    raise RuntimeError("Downloaded MeshLab archive does not contain meshlab.exe")


def _write_manifest(result: MeshLabBinaryResult) -> None:
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


def download_meshlab_for_windows(*, log: LogFn = print) -> MeshLabBinaryResult:
    if os.name != "nt":
        raise RuntimeError("Automatic MeshLab download is currently implemented for Windows only.")

    runtime_root = _runtime_tools_root()
    runtime_root.mkdir(parents=True, exist_ok=True)

    release_json = request_json(MESHLAB_RELEASE_API_URL)
    asset_name, download_url, tag_name = _choose_windows_asset(release_json)
    final_root = runtime_root / (tag_name or "latest")
    final_binary = final_root / "meshlab.exe"
    if final_binary.exists():
        result = MeshLabBinaryResult(
            binary_path=str(final_binary.resolve()),
            source="runtime_cached",
            install_dir=str(final_root.resolve()),
            release_tag=tag_name or None,
            asset_name=asset_name,
        )
        _write_manifest(result)
        return result

    log(f"[INFO] Downloading official MeshLab release {tag_name or '(latest)'}: {asset_name}")
    archive_path = runtime_root / asset_name
    staging_dir = runtime_root / "_staging"

    if archive_path.exists():
        archive_path.unlink()
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    download_file_with_resume(download_url, archive_path, label="MeshLab", log=log)
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(staging_dir)

    extracted_dir = _find_extracted_meshlab_dir(staging_dir)
    if final_root.exists():
        shutil.rmtree(final_root)
    final_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(extracted_dir), str(final_root))

    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    if archive_path.exists():
        archive_path.unlink()

    result = MeshLabBinaryResult(
        binary_path=str((final_root / "meshlab.exe").resolve()),
        source="runtime_downloaded",
        install_dir=str(final_root.resolve()),
        release_tag=tag_name or None,
        asset_name=asset_name,
    )
    _write_manifest(result)
    return result


def ensure_meshlab_binary(
    *,
    preferred_path: str | None = None,
    download_if_missing: bool = False,
    log: LogFn = print,
) -> MeshLabBinaryResult:
    detected = detect_meshlab_binary(preferred_path=preferred_path)
    if detected is not None:
        return detected
    if download_if_missing:
        return download_meshlab_for_windows(log=log)
    raise FileNotFoundError("找不到 MeshLab。请安装 MeshLab，或允许 GUI 自动下载官方 Windows 版 MeshLab。")


def open_point_cloud_in_meshlab(
    *,
    point_cloud_path: str,
    preferred_path: str | None = None,
    download_if_missing: bool = False,
    log: LogFn = print,
) -> MeshLabBinaryResult:
    point_cloud = Path(point_cloud_path).expanduser().resolve()
    if not point_cloud.exists():
        raise FileNotFoundError(f"Point cloud file not found: {point_cloud}")

    binary = ensure_meshlab_binary(preferred_path=preferred_path, download_if_missing=download_if_missing, log=log)
    subprocess.Popen([binary.binary_path, str(point_cloud)], cwd=str(point_cloud.parent))
    log(f"[OK] Opened point cloud in MeshLab: {point_cloud}")
    return binary
