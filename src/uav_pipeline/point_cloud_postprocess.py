from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

LogFn = Callable[[str], None]

_PLY_DTYPES = {
    "char": "i1",
    "uchar": "u1",
    "short": "i2",
    "ushort": "u2",
    "int": "i4",
    "uint": "u4",
    "float": "f4",
    "double": "f8",
}


@dataclass(frozen=True)
class PointCloudPostprocessResult:
    output_path: str
    raw_backup_path: str
    metadata_path: str
    input_points: int
    output_points: int
    center: tuple[float, float, float]
    trim_percent: float


def _read_binary_ply_vertices(path: Path) -> tuple[bytes, np.ndarray]:
    with path.open("rb") as f:
        header_lines: list[bytes] = []
        fmt: str | None = None
        vertex_count: int | None = None
        vertex_props: list[tuple[str, str]] = []
        current_element: str | None = None

        while True:
            line = f.readline()
            if not line:
                raise RuntimeError(f"PLY header is incomplete: {path}")
            header_lines.append(line)
            text = line.decode("ascii", errors="strict").strip()
            if text == "end_header":
                break
            if not text or text == "ply" or text.startswith("comment"):
                continue
            if text.startswith("format "):
                parts = text.split()
                if len(parts) < 3:
                    raise RuntimeError(f"Invalid PLY format header: {path}")
                fmt = parts[1]
                continue
            if text.startswith("element "):
                parts = text.split()
                if len(parts) != 3:
                    raise RuntimeError(f"Invalid PLY element header: {path}")
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
                elif vertex_count is not None:
                    raise RuntimeError(f"Unsupported extra PLY element after vertex: {current_element}")
                continue
            if text.startswith("property "):
                parts = text.split()
                if len(parts) != 3 or current_element != "vertex":
                    raise RuntimeError(f"Unsupported PLY property layout: {path}")
                ptype, pname = parts[1], parts[2]
                if ptype not in _PLY_DTYPES:
                    raise RuntimeError(f"Unsupported PLY property type {ptype} in {path}")
                vertex_props.append((pname, ptype))
                continue

        if fmt != "binary_little_endian":
            raise RuntimeError(f"Only binary_little_endian PLY is supported, got {fmt!r} in {path}")
        if vertex_count is None or not vertex_props:
            raise RuntimeError(f"PLY vertex layout is missing: {path}")

        dtype = np.dtype([(name, "<" + _PLY_DTYPES[ptype]) for name, ptype in vertex_props])
        vertices = np.fromfile(f, dtype=dtype, count=vertex_count)
        if len(vertices) != vertex_count:
            raise RuntimeError(f"Expected {vertex_count} PLY vertices, got {len(vertices)} in {path}")
        return b"".join(header_lines), vertices


def _write_binary_ply_vertices(path: Path, header: bytes, vertices: np.ndarray) -> None:
    header_lines = header.decode("ascii", errors="strict").splitlines()
    updated_lines: list[str] = []
    for line in header_lines:
        if line.startswith("element vertex "):
            updated_lines.append(f"element vertex {len(vertices)}")
        else:
            updated_lines.append(line)
    updated_header = ("\n".join(updated_lines) + "\n").encode("ascii")

    with path.open("wb") as f:
        f.write(updated_header)
        vertices.tofile(f)


def postprocess_point_cloud(
    *,
    fused_path: str,
    trim_percent: float = 0.5,
    log: LogFn = print,
) -> PointCloudPostprocessResult:
    if trim_percent < 0 or trim_percent >= 50:
        raise ValueError("trim_percent 必须在 [0, 50) 范围内。")

    output_path = Path(fused_path).expanduser().resolve()
    if not output_path.exists():
        raise FileNotFoundError(f"找不到点云文件: {output_path}")

    raw_backup_path = output_path.with_name(output_path.stem + "_raw" + output_path.suffix)
    metadata_path = output_path.with_name(output_path.stem + "_postprocess.json")

    header, vertices = _read_binary_ply_vertices(output_path)
    for name in ("x", "y", "z"):
        if name not in vertices.dtype.names:
            raise RuntimeError(f"PLY 缺少 {name} 坐标字段: {output_path}")

    input_points = int(len(vertices))
    if input_points == 0:
        raise RuntimeError(f"点云为空: {output_path}")

    points = np.column_stack([vertices["x"], vertices["y"], vertices["z"]]).astype(np.float64, copy=False)

    mask = np.ones(input_points, dtype=bool)
    if trim_percent > 0 and input_points >= 100:
        low = trim_percent
        high = 100.0 - trim_percent
        lo = np.percentile(points, low, axis=0)
        hi = np.percentile(points, high, axis=0)
        candidate_mask = np.all((points >= lo) & (points <= hi), axis=1)
        kept = int(candidate_mask.sum())
        if kept >= max(100, int(input_points * 0.5)):
            mask = candidate_mask
            log(f"[INFO] Point cloud trim kept {kept}/{input_points} points at {trim_percent}% per-axis trim")
        else:
            log("[WARN] Point cloud trim would remove too many points; keeping all points")

    filtered = vertices[mask].copy()
    filtered_points = np.column_stack([filtered["x"], filtered["y"], filtered["z"]]).astype(np.float64, copy=False)
    bbox_min = filtered_points.min(axis=0)
    bbox_max = filtered_points.max(axis=0)
    center = (bbox_min + bbox_max) / 2.0

    filtered["x"] = (filtered_points[:, 0] - center[0]).astype(filtered.dtype["x"])
    filtered["y"] = (filtered_points[:, 1] - center[1]).astype(filtered.dtype["y"])
    filtered["z"] = (filtered_points[:, 2] - center[2]).astype(filtered.dtype["z"])

    shutil.copy2(output_path, raw_backup_path)
    _write_binary_ply_vertices(output_path, header, filtered)

    metadata = {
        "input_path": str(output_path),
        "raw_backup_path": str(raw_backup_path),
        "input_points": input_points,
        "output_points": int(len(filtered)),
        "trim_percent": trim_percent,
        "center_mode": "bbox",
        "center": center.tolist(),
        "bbox_min_before_centering": bbox_min.tolist(),
        "bbox_max_before_centering": bbox_max.tolist(),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

    log(f"[OK] Postprocessed point cloud: {output_path}")
    return PointCloudPostprocessResult(
        output_path=str(output_path),
        raw_backup_path=str(raw_backup_path),
        metadata_path=str(metadata_path),
        input_points=input_points,
        output_points=int(len(filtered)),
        center=(float(center[0]), float(center[1]), float(center[2])),
        trim_percent=trim_percent,
    )
