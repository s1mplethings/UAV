from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2

LogFn = Callable[[str], None]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
MANIFEST_NAME = "video_input.json"
SELECTION_VERSION = "smart-v2"
PREVIEW_WIDTH = 160


@dataclass(frozen=True)
class VideoInputResult:
    images_dir: str
    manifest_path: str
    extracted_frames: int
    source_fps: float | None
    reused: bool


@dataclass
class _VideoFrameCandidate:
    frame_index: int
    timestamp_sec: float
    blur_score: float
    brightness: float
    preview_gray: object


def _image_files(images_dir: Path) -> list[Path]:
    if not images_dir.exists():
        return []
    return sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _load_manifest(manifest_path: Path) -> dict[str, object] | None:
    if not manifest_path.exists():
        return None
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _manifest_matches(
    manifest: dict[str, object],
    *,
    video_path: Path,
    sample_fps: float,
    max_frames: int | None,
    blur_threshold: float,
    dedupe_threshold: float,
    min_gap_sec: float,
    image_count: int,
) -> bool:
    stat = video_path.stat()
    return (
        manifest.get("selection_version") == SELECTION_VERSION
        and manifest.get("video_path") == str(video_path)
        and manifest.get("video_size") == stat.st_size
        and manifest.get("video_mtime_ns") == stat.st_mtime_ns
        and manifest.get("sample_fps") == sample_fps
        and manifest.get("max_frames") == max_frames
        and manifest.get("blur_threshold") == blur_threshold
        and manifest.get("dedupe_threshold") == dedupe_threshold
        and manifest.get("min_gap_sec") == min_gap_sec
        and manifest.get("extracted_frames") == image_count
    )


def _build_preview(gray_frame):
    height, width = gray_frame.shape[:2]
    target_width = min(PREVIEW_WIDTH, width)
    if target_width == width:
        return gray_frame
    target_height = max(1, round(height * target_width / width))
    return cv2.resize(gray_frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _sample_candidates(
    *,
    source_video: Path,
    sample_fps: float,
    log: LogFn,
) -> tuple[list[_VideoFrameCandidate], float | None]:
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频文件: {source_video}")

    source_fps_raw = float(capture.get(cv2.CAP_PROP_FPS))
    source_fps = source_fps_raw if source_fps_raw > 0 else None
    if source_fps is not None:
        frame_interval = max(source_fps / sample_fps, 1.0)
        log(f"[INFO] Video FPS: {source_fps:.3f}; sampling at {sample_fps:.3f} FPS")
    else:
        frame_interval = 1.0
        log("[WARN] Unable to read source video FPS; falling back to sequential frame extraction.")

    candidates: list[_VideoFrameCandidate] = []
    frame_idx = 0
    next_frame_idx = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            should_sample = source_fps is None or frame_idx + 1e-9 >= next_frame_idx
            if should_sample:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                preview = _build_preview(gray)
                blur_score = float(cv2.Laplacian(preview, cv2.CV_64F).var())
                brightness = float(preview.mean())
                timestamp_sec = (frame_idx / source_fps) if source_fps else float(frame_idx)
                candidates.append(
                    _VideoFrameCandidate(
                        frame_index=frame_idx,
                        timestamp_sec=timestamp_sec,
                        blur_score=blur_score,
                        brightness=brightness,
                        preview_gray=preview,
                    )
                )
                if source_fps is not None:
                    next_frame_idx += frame_interval

            frame_idx += 1
    finally:
        capture.release()

    return candidates, source_fps


def _quality_score(candidate: _VideoFrameCandidate) -> float:
    brightness_penalty = abs(candidate.brightness - 128.0) * 1.5
    return candidate.blur_score - brightness_penalty


def _filter_candidates(
    candidates: list[_VideoFrameCandidate],
    *,
    blur_threshold: float,
    dedupe_threshold: float,
    min_gap_sec: float,
) -> tuple[list[_VideoFrameCandidate], dict[str, int]]:
    kept: list[_VideoFrameCandidate] = []
    rejected_blur = 0
    rejected_dedupe = 0
    rejected_gap = 0
    last_kept: _VideoFrameCandidate | None = None

    for candidate in candidates:
        if blur_threshold > 0 and candidate.blur_score < blur_threshold:
            rejected_blur += 1
            continue
        if min_gap_sec > 0 and last_kept is not None:
            if candidate.timestamp_sec - last_kept.timestamp_sec < min_gap_sec:
                rejected_gap += 1
                continue
        if dedupe_threshold > 0 and last_kept is not None:
            diff = float(cv2.absdiff(candidate.preview_gray, last_kept.preview_gray).mean())
            if diff < dedupe_threshold:
                rejected_dedupe += 1
                continue
        kept.append(candidate)
        last_kept = candidate

    if kept:
        return kept, {
            "rejected_blur": rejected_blur,
            "rejected_dedupe": rejected_dedupe,
            "rejected_gap": rejected_gap,
        }

    fallback = [max(candidates, key=_quality_score)] if candidates else []
    return fallback, {
        "rejected_blur": rejected_blur,
        "rejected_dedupe": rejected_dedupe,
        "rejected_gap": rejected_gap,
    }


def _downselect_candidates(candidates: list[_VideoFrameCandidate], max_frames: int | None) -> list[_VideoFrameCandidate]:
    if max_frames is None or len(candidates) <= max_frames:
        return candidates

    total = len(candidates)
    picks: list[_VideoFrameCandidate] = []
    picked_indices: set[int] = set()
    for slot in range(max_frames):
        start = slot * total // max_frames
        end = (slot + 1) * total // max_frames
        if end <= start:
            end = start + 1
        bucket = candidates[start:end]
        chosen = max(bucket, key=_quality_score)
        if chosen.frame_index in picked_indices:
            continue
        picks.append(chosen)
        picked_indices.add(chosen.frame_index)

    if len(picks) < max_frames:
        remaining = sorted(
            (candidate for candidate in candidates if candidate.frame_index not in picked_indices),
            key=_quality_score,
            reverse=True,
        )
        for candidate in remaining:
            picks.append(candidate)
            picked_indices.add(candidate.frame_index)
            if len(picks) >= max_frames:
                break

    return sorted(picks, key=lambda candidate: candidate.frame_index)


def _write_selected_frames(
    *,
    source_video: Path,
    images_dir: Path,
    selected_candidates: list[_VideoFrameCandidate],
) -> None:
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        raise RuntimeError(f"无法重新打开视频文件: {source_video}")

    target_ptr = 0
    frame_idx = 0
    try:
        while target_ptr < len(selected_candidates):
            ok, frame = capture.read()
            if not ok:
                break
            target = selected_candidates[target_ptr]
            if frame_idx == target.frame_index:
                out_path = images_dir / f"frame_{target_ptr + 1:04d}.jpg"
                if not cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
                    raise RuntimeError(f"写入抽帧文件失败: {out_path}")
                target_ptr += 1
            frame_idx += 1
    finally:
        capture.release()

    if target_ptr != len(selected_candidates):
        raise RuntimeError("视频二次读取时未能写出全部选定帧。")


def prepare_work_dir_from_video(
    *,
    work_dir: str,
    video_path: str | None,
    sample_fps: float = 2.0,
    max_frames: int | None = None,
    blur_threshold: float = 0.0,
    dedupe_threshold: float = 0.0,
    min_gap_sec: float = 0.0,
    overwrite: bool = False,
    log: LogFn = print,
) -> VideoInputResult | None:
    if not video_path:
        return None

    if sample_fps <= 0:
        raise ValueError("video_sample_fps 必须大于 0。")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("video_max_frames 必须是正整数或留空。")
    if blur_threshold < 0:
        raise ValueError("video_blur_threshold 不能小于 0。")
    if dedupe_threshold < 0:
        raise ValueError("video_dedupe_threshold 不能小于 0。")
    if min_gap_sec < 0:
        raise ValueError("video_min_gap_sec 不能小于 0。")

    work_root = Path(work_dir).expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    source_video = Path(video_path).expanduser().resolve()
    if not source_video.exists():
        raise FileNotFoundError(f"找不到视频文件: {source_video}")
    if not source_video.is_file():
        raise FileNotFoundError(f"视频路径不是文件: {source_video}")

    images_dir = work_root / "images"
    manifest_path = work_root / MANIFEST_NAME

    existing_images = _image_files(images_dir)
    if existing_images:
        manifest = _load_manifest(manifest_path)
        if not overwrite and manifest and _manifest_matches(
            manifest,
            video_path=source_video,
            sample_fps=sample_fps,
            max_frames=max_frames,
            blur_threshold=blur_threshold,
            dedupe_threshold=dedupe_threshold,
            min_gap_sec=min_gap_sec,
            image_count=len(existing_images),
        ):
            log(f"[INFO] Reusing extracted video frames in: {images_dir}")
            return VideoInputResult(
                images_dir=str(images_dir),
                manifest_path=str(manifest_path),
                extracted_frames=int(manifest.get("extracted_frames", len(existing_images))),
                source_fps=float(manifest["source_fps"]) if manifest.get("source_fps") is not None else None,
                reused=True,
            )
        if not overwrite:
            raise RuntimeError(
                f"{images_dir} 已存在图片。若要用视频重建，请换一个空的工作目录，或加 --overwrite 允许重建 images/。"
            )
    elif images_dir.exists() and any(images_dir.iterdir()) and not overwrite:
        raise RuntimeError(
            f"{images_dir} 已存在内容，但不是可复用的视频抽帧结果。请清空该目录，或加 --overwrite 后重试。"
        )

    if images_dir.exists() and overwrite:
        log(f"[INFO] Removing existing extracted frames: {images_dir}")
        shutil.rmtree(images_dir)

    images_dir.mkdir(parents=True, exist_ok=True)

    log(f"[INFO] Extracting frames from video: {source_video}")
    sampled_candidates, source_fps = _sample_candidates(source_video=source_video, sample_fps=sample_fps, log=log)
    if not sampled_candidates:
        raise RuntimeError(f"没有从视频中提取到任何帧: {source_video}")

    filtered_candidates, filter_stats = _filter_candidates(
        sampled_candidates,
        blur_threshold=blur_threshold,
        dedupe_threshold=dedupe_threshold,
        min_gap_sec=min_gap_sec,
    )
    selected_candidates = _downselect_candidates(filtered_candidates, max_frames=max_frames)

    log(
        "[INFO] Video frame selection: "
        f"sampled={len(sampled_candidates)}, "
        f"kept_after_filters={len(filtered_candidates)}, "
        f"selected={len(selected_candidates)}"
    )
    if any(filter_stats.values()):
        log(
            "[INFO] Filter rejects: "
            f"blur={filter_stats['rejected_blur']}, "
            f"dedupe={filter_stats['rejected_dedupe']}, "
            f"min_gap={filter_stats['rejected_gap']}"
        )
    if filtered_candidates and not selected_candidates:
        raise RuntimeError("视频抽帧筛选后没有可写出的帧。")
    if not filtered_candidates and sampled_candidates:
        log("[WARN] Video filters removed all sampled frames; falling back to the sharpest sampled frame.")
        selected_candidates = _downselect_candidates([max(sampled_candidates, key=_quality_score)], max_frames=max_frames)

    _write_selected_frames(
        source_video=source_video,
        images_dir=images_dir,
        selected_candidates=selected_candidates,
    )

    written = len(selected_candidates)
    stat = source_video.stat()
    manifest = {
        "selection_version": SELECTION_VERSION,
        "video_path": str(source_video),
        "video_size": stat.st_size,
        "video_mtime_ns": stat.st_mtime_ns,
        "sample_fps": sample_fps,
        "max_frames": max_frames,
        "blur_threshold": blur_threshold,
        "dedupe_threshold": dedupe_threshold,
        "min_gap_sec": min_gap_sec,
        "source_fps": source_fps,
        "sampled_candidates": len(sampled_candidates),
        "filtered_candidates": len(filtered_candidates),
        "extracted_frames": written,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")

    log(f"[OK] Extracted {written} frame(s) to: {images_dir}")
    return VideoInputResult(
        images_dir=str(images_dir),
        manifest_path=str(manifest_path),
        extracted_frames=written,
        source_fps=source_fps,
        reused=False,
    )
