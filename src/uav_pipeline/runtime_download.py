from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

LogFn = Callable[[str], None]

DEFAULT_RUNTIME_USER_AGENT = "uav-workbench-gui/1.0"
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_PROGRESS_STEP = 10 * 1024 * 1024


def request_json(url: str, *, timeout: int = 60) -> dict[str, object]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": DEFAULT_RUNTIME_USER_AGENT,
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _total_bytes(response: urllib.response.addinfourl, resumed_from: int) -> int:
    content_range = response.headers.get("Content-Range", "")
    if "/" in content_range:
        total_raw = content_range.rsplit("/", 1)[-1].strip()
        if total_raw.isdigit():
            return int(total_raw)
    content_length = response.headers.get("Content-Length", "")
    if content_length.isdigit():
        return resumed_from + int(content_length)
    return 0


def _download_with_curl(
    url: str,
    destination: Path,
    *,
    label: str,
    log: LogFn,
) -> bool:
    curl_path = shutil.which("curl.exe") or shutil.which("curl")
    if not curl_path:
        return False
    log(f"[INFO] Falling back to system curl for {label} download.")
    command = [
        curl_path,
        "--location",
        "--fail",
        "--retry",
        "4",
        "--retry-all-errors",
        "--retry-delay",
        "2",
        "--connect-timeout",
        "30",
        "--continue-at",
        "-",
        "--user-agent",
        DEFAULT_RUNTIME_USER_AGENT,
        "--output",
        str(destination),
        url,
    ]
    try:
        proc = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60 * 30,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        log(f"[WARN] curl fallback failed to start for {label}: {exc}")
        return False
    if proc.returncode == 0:
        log(f"[OK] {label} download completed with system curl.")
        return True
    stderr = (proc.stderr or "").strip()
    log(f"[WARN] curl fallback failed for {label}: {stderr or f'return code {proc.returncode}'}")
    return False


def download_file_with_resume(
    url: str,
    destination: Path,
    *,
    label: str,
    log: LogFn,
    timeout: int = 120,
    max_attempts: int = 4,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    next_report = 0
    total_bytes = 0
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        resumed_from = destination.stat().st_size if destination.exists() else 0
        headers = {"User-Agent": DEFAULT_RUNTIME_USER_AGENT}
        if resumed_from > 0:
            headers["Range"] = f"bytes={resumed_from}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_code = response.getcode()
                if resumed_from > 0 and response_code != 206:
                    log(f"[WARN] {label} server did not honor resume; restarting download from 0 MB.")
                    destination.unlink(missing_ok=True)
                    resumed_from = 0

                total_bytes = _total_bytes(response, resumed_from)
                downloaded = resumed_from
                if downloaded >= next_report:
                    next_report = downloaded + DOWNLOAD_PROGRESS_STEP

                with destination.open("ab" if resumed_from > 0 else "wb") as handle:
                    while True:
                        chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if total_bytes and downloaded >= next_report:
                            percent = downloaded * 100.0 / total_bytes
                            log(
                                f"[INFO] Downloading {label}: {downloaded / 1024 / 1024:.1f} / "
                                f"{total_bytes / 1024 / 1024:.1f} MB ({percent:.1f}%)"
                            )
                            next_report = downloaded + DOWNLOAD_PROGRESS_STEP

                if total_bytes and downloaded < total_bytes:
                    raise IOError(f"incomplete download: expected {total_bytes} bytes, got {downloaded} bytes")
                return
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and destination.exists():
                return
            last_error = exc
            if attempt == max_attempts:
                break
            log(f"[WARN] {label} download failed ({attempt}/{max_attempts}): {exc}. Retrying...")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt == max_attempts:
                break
            log(f"[WARN] {label} download interrupted ({attempt}/{max_attempts}): {exc}. Retrying...")
        time.sleep(min(2 * attempt, 6))

    if _download_with_curl(url, destination, label=label, log=log):
        return

    if total_bytes and destination.exists() and destination.stat().st_size != total_bytes:
        raise IOError(
            f"{label} download incomplete after retries: expected {total_bytes} bytes, got {destination.stat().st_size} bytes"
        )
    if last_error is not None:
        raise last_error
