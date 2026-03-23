from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

LogFn = Callable[[str], None]

DEFAULT_POINT_CLOUD_API_VIEWS: tuple[tuple[str, float, float], ...] = (
    ("Front", 0.0, 0.0),
    ("Right", 90.0, 0.0),
    ("Top", 0.0, -90.0),
    ("Isometric", 35.0, -25.0),
)
POINT_CLOUD_RENDER_STYLES: tuple[str, ...] = ("points", "surface")

_PLY_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


@dataclass
class PointCloudRenderData:
    point_cloud_path: str
    aligned_points: np.ndarray
    colors: np.ndarray | None
    input_points: int
    rendered_points: int


@dataclass(frozen=True)
class PointCloudViewResult:
    point_cloud_path: str
    image_path: str
    metadata_path: str
    title: str
    render_style: str
    input_points: int
    rendered_points: int
    width: int
    height: int
    yaw_deg: float
    pitch_deg: float
    pan_x: float
    pan_y: float
    zoom: float


@dataclass(frozen=True)
class PointCloudViewSetResult:
    point_cloud_path: str
    output_dir: str
    metadata_path: str
    render_style: str
    input_points: int
    rendered_points: int
    view_names: tuple[str, ...]
    image_paths: tuple[str, ...]


@dataclass(frozen=True)
class PointCloudSnapshotResult:
    point_cloud_path: str
    image_path: str
    metadata_path: str
    render_style: str
    input_points: int
    rendered_points: int
    width: int
    height: int
    view_names: tuple[str, ...]


def _decode_header_line(raw_line: bytes) -> str:
    return raw_line.decode("ascii", errors="strict").strip()


def _load_ply_vertices(ply_path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with ply_path.open("rb") as handle:
        format_name: str | None = None
        vertex_count: int | None = None
        properties: list[tuple[str, str]] = []
        in_vertex = False

        while True:
            raw_line = handle.readline()
            if not raw_line:
                raise RuntimeError(f"PLY header not terminated: {ply_path}")
            line = _decode_header_line(raw_line)
            if line.startswith("format "):
                parts = line.split()
                if len(parts) >= 2:
                    format_name = parts[1]
            elif line.startswith("element "):
                parts = line.split()
                in_vertex = len(parts) >= 3 and parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
                    properties = []
            elif line.startswith("property ") and in_vertex:
                parts = line.split()
                if len(parts) >= 3 and parts[1] != "list":
                    properties.append((parts[2], parts[1]))
            elif line == "end_header":
                break

        if format_name is None:
            raise RuntimeError(f"PLY format missing: {ply_path}")
        if vertex_count is None:
            raise RuntimeError(f"PLY vertex count missing: {ply_path}")

        property_names = [name for name, _ in properties]
        if not {"x", "y", "z"}.issubset(property_names):
            raise RuntimeError(f"PLY missing x/y/z vertex fields: {ply_path}")

        if format_name == "ascii":
            data = np.loadtxt(handle, max_rows=vertex_count)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            points = data[:, [property_names.index("x"), property_names.index("y"), property_names.index("z")]]
            colors = None
            if {"red", "green", "blue"}.issubset(property_names):
                colors = data[
                    :,
                    [property_names.index("red"), property_names.index("green"), property_names.index("blue")],
                ]
            return points.astype(np.float32), colors.astype(np.uint8) if colors is not None else None

        if format_name != "binary_little_endian":
            raise RuntimeError(f"Unsupported PLY format: {format_name}")

        dtype_fields = []
        for name, ply_type in properties:
            np_type = _PLY_DTYPES.get(ply_type)
            if np_type is None:
                raise RuntimeError(f"Unsupported PLY property type {ply_type!r} in {ply_path}")
            dtype_fields.append((name, "<" + np_type))

        vertices = np.fromfile(handle, dtype=np.dtype(dtype_fields), count=vertex_count)
        points = np.stack([vertices["x"], vertices["y"], vertices["z"]], axis=1).astype(np.float32)

        colors = None
        if {"red", "green", "blue"}.issubset(property_names):
            colors = np.stack([vertices["red"], vertices["green"], vertices["blue"]], axis=1).astype(np.uint8)
        return points, colors


def _subsample_points(
    points: np.ndarray,
    colors: np.ndarray | None,
    *,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if len(points) <= max_points:
        return points, colors
    indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int32)
    return points[indices], colors[indices] if colors is not None else None


def _principal_axes(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis = vh.T
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    return basis.astype(np.float32)


def _rotate_points(points: np.ndarray, *, yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    yaw_rot = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    pitch_rot = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=np.float32,
    )
    return points @ yaw_rot.T @ pitch_rot.T


def _depth_colors(depth: np.ndarray) -> np.ndarray:
    if len(depth) == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    depth_min = float(depth.min())
    depth_max = float(depth.max())
    if depth_max <= depth_min + 1e-6:
        grayscale = np.full((len(depth),), 180, dtype=np.uint8)
    else:
        grayscale = ((depth - depth_min) / (depth_max - depth_min) * 255.0).astype(np.uint8)
    mapped = cv2.applyColorMap(grayscale.reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
    return mapped.astype(np.uint8)


def _project_points(
    points: np.ndarray,
    colors: np.ndarray | None,
    *,
    width: int,
    height: int,
    padding: int = 24,
    pan_x: float = 0.0,
    pan_y: float = 0.0,
    zoom: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(points) == 0:
        return (
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
        )

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    max_span = max(float(x.max() - x.min()), float(y.max() - y.min()), 1e-6)
    scale = min((width - 2 * padding) / max_span, (height - 2 * padding) / max_span) * max(float(zoom), 1e-3)
    usable_width = max(width - 2 * padding, 1)
    usable_height = max(height - 2 * padding, 1)
    x_center = float((x.max() + x.min()) * 0.5)
    y_center = float((y.max() + y.min()) * 0.5)

    px = ((x - x_center) * scale + width * 0.5 + float(pan_x) * usable_width * 0.5).astype(np.int32)
    py = (height * 0.5 - (y - y_center) * scale + float(pan_y) * usable_height * 0.5).astype(np.int32)
    valid = (px >= padding) & (px < width - padding) & (py >= padding) & (py < height - padding)
    px = px[valid]
    py = py[valid]
    depth = z[valid]
    if colors is not None:
        draw_colors = colors[valid][:, ::-1]
    else:
        draw_colors = _depth_colors(depth)
    return px, py, depth, draw_colors


def _surface_radius(point_count: int, width: int, height: int) -> int:
    density = max(point_count / max(width * height, 1), 1e-6)
    if density < 0.0015:
        return 6
    if density < 0.004:
        return 5
    if density < 0.008:
        return 4
    if density < 0.015:
        return 3
    return 2


def _render_points_view(
    *,
    canvas: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    depth: np.ndarray,
    draw_colors: np.ndarray,
) -> np.ndarray:
    if len(px) == 0:
        return canvas

    order = np.argsort(depth)
    radius = 1 if len(order) > 15000 else 2
    for idx in order:
        color = tuple(int(channel) for channel in draw_colors[idx])
        cv2.circle(canvas, (int(px[idx]), int(py[idx])), radius, color, thickness=-1, lineType=cv2.LINE_AA)
    return canvas


def _render_surface_view(
    *,
    canvas: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    depth: np.ndarray,
    draw_colors: np.ndarray,
) -> np.ndarray:
    if len(px) == 0:
        return canvas

    radius = _surface_radius(len(px), canvas.shape[1], canvas.shape[0])
    order = np.argsort(depth)
    mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
    color_layer = np.zeros_like(canvas)
    depth_layer = np.zeros(canvas.shape[:2], dtype=np.uint8)
    depth_min = float(depth.min())
    depth_span = max(float(depth.max() - depth_min), 1e-6)
    depth_uint8 = ((depth - depth_min) / depth_span * 255.0).astype(np.uint8)

    for idx in order:
        point = (int(px[idx]), int(py[idx]))
        color = tuple(int(channel) for channel in draw_colors[idx])
        cv2.circle(mask, point, radius, 255, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(color_layer, point, radius, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(depth_layer, point, radius, int(depth_uint8[idx]), thickness=-1, lineType=cv2.LINE_AA)

    kernel_size = radius * 3 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    filled_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    fill_only = cv2.subtract(filled_mask, mask)
    color_filled = cv2.inpaint(color_layer, fill_only, max(2, radius), cv2.INPAINT_NS)
    depth_filled = cv2.inpaint(cv2.cvtColor(depth_layer, cv2.COLOR_GRAY2BGR), fill_only, max(2, radius), cv2.INPAINT_NS)
    depth_gray = cv2.cvtColor(depth_filled, cv2.COLOR_BGR2GRAY)
    soft_mask = cv2.GaussianBlur(filled_mask, (0, 0), sigmaX=max(0.8, radius * 0.7)).astype(np.float32) / 255.0
    color_smoothed = cv2.GaussianBlur(color_filled, (0, 0), sigmaX=max(0.8, radius * 0.6))
    shade = 0.82 + 0.22 * (1.0 - depth_gray.astype(np.float32) / 255.0)
    shaded = np.clip(color_smoothed.astype(np.float32) * shade[..., None], 0, 255).astype(np.uint8)
    composite = canvas.astype(np.float32) * (1.0 - soft_mask[..., None]) + shaded.astype(np.float32) * soft_mask[..., None]
    composite = composite.astype(np.uint8)
    edges = cv2.Canny(filled_mask, 40, 120)
    composite[edges > 0] = np.clip(composite[edges > 0].astype(np.float32) * 0.78, 0, 255).astype(np.uint8)
    return composite


def _render_view(
    points: np.ndarray,
    colors: np.ndarray | None,
    *,
    title: str,
    width: int,
    height: int,
    render_style: str,
    padding: int = 24,
    pan_x: float = 0.0,
    pan_y: float = 0.0,
    zoom: float = 1.0,
) -> np.ndarray:
    if render_style not in POINT_CLOUD_RENDER_STYLES:
        raise ValueError(f"Unsupported render_style: {render_style}")

    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    px, py, depth, draw_colors = _project_points(
        points,
        colors,
        width=width,
        height=height,
        padding=padding,
        pan_x=pan_x,
        pan_y=pan_y,
        zoom=zoom,
    )
    if len(px) == 0:
        cv2.putText(canvas, f"{title}: empty", (padding, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        return canvas

    if render_style == "surface":
        canvas = _render_surface_view(canvas=canvas, px=px, py=py, depth=depth, draw_colors=draw_colors)
    else:
        canvas = _render_points_view(canvas=canvas, px=px, py=py, depth=depth, draw_colors=draw_colors)

    cv2.putText(canvas, title, (padding, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (32, 32, 32), 2, cv2.LINE_AA)
    return canvas


def _slugify_view_name(name: str) -> str:
    slug = name.strip().lower().replace(" ", "_")
    return "".join(ch for ch in slug if ch.isalnum() or ch == "_") or "view"


def _resolve_render_data(
    *,
    point_cloud_path: str | None,
    render_data: PointCloudRenderData | None,
    max_points: int,
) -> PointCloudRenderData:
    if render_data is not None:
        return render_data
    if not point_cloud_path:
        raise ValueError("point_cloud_path is required when render_data is not provided.")
    return prepare_point_cloud_render_data(point_cloud_path=point_cloud_path, max_points=max_points)


def _compose_panel_grid(panels: Sequence[np.ndarray], *, columns: int = 2, gap: int = 12) -> np.ndarray:
    if not panels:
        raise ValueError("panels must not be empty")
    rows = int(math.ceil(len(panels) / columns))
    panel_height, panel_width = panels[0].shape[:2]
    background = np.full(
        (rows * panel_height + (rows - 1) * gap, columns * panel_width + (columns - 1) * gap, 3),
        255,
        dtype=np.uint8,
    )
    for index, panel in enumerate(panels):
        row = index // columns
        column = index % columns
        top = row * (panel_height + gap)
        left = column * (panel_width + gap)
        background[top : top + panel_height, left : left + panel_width] = panel
    return background


def prepare_point_cloud_render_data(
    *,
    point_cloud_path: str,
    max_points: int = 45000,
) -> PointCloudRenderData:
    ply_path = Path(point_cloud_path).expanduser().resolve()
    if not ply_path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {ply_path}")

    points, colors = _load_ply_vertices(ply_path)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if colors is not None:
        colors = colors[finite]
    if len(points) == 0:
        raise RuntimeError(f"Point cloud has no valid points: {ply_path}")

    sampled_points, sampled_colors = _subsample_points(points, colors, max_points=max_points)
    basis = _principal_axes(sampled_points)
    aligned = (sampled_points - sampled_points.mean(axis=0, keepdims=True)) @ basis
    return PointCloudRenderData(
        point_cloud_path=str(ply_path),
        aligned_points=aligned.astype(np.float32),
        colors=sampled_colors,
        input_points=int(len(points)),
        rendered_points=int(len(sampled_points)),
    )


def render_point_cloud_view(
    *,
    point_cloud_path: str | None = None,
    render_data: PointCloudRenderData | None = None,
    output_path: str | None = None,
    title: str = "Preview",
    render_style: str = "points",
    yaw_deg: float = 35.0,
    pitch_deg: float = -25.0,
    pan_x: float = 0.0,
    pan_y: float = 0.0,
    zoom: float = 1.0,
    width: int = 520,
    height: int = 520,
    max_points: int = 45000,
    log: LogFn = print,
) -> PointCloudViewResult:
    prepared = _resolve_render_data(point_cloud_path=point_cloud_path, render_data=render_data, max_points=max_points)
    point_cloud = Path(prepared.point_cloud_path)
    image_path = (
        Path(output_path).expanduser().resolve()
        if output_path
        else point_cloud.with_name(f"{_slugify_view_name(title)}.png")
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)

    rotated = _rotate_points(prepared.aligned_points, yaw_deg=yaw_deg, pitch_deg=pitch_deg)
    image = _render_view(
        rotated,
        prepared.colors,
        title=title,
        width=width,
        height=height,
        render_style=render_style,
        pan_x=pan_x,
        pan_y=pan_y,
        zoom=zoom,
    )
    if not cv2.imwrite(str(image_path), image):
        raise RuntimeError(f"Failed to write point-cloud view image: {image_path}")

    metadata_path = image_path.with_suffix(".json")
    metadata = {
        "point_cloud_path": prepared.point_cloud_path,
        "image_path": str(image_path),
        "title": title,
        "render_style": render_style,
        "yaw_deg": float(yaw_deg),
        "pitch_deg": float(pitch_deg),
        "pan_x": float(pan_x),
        "pan_y": float(pan_y),
        "zoom": float(zoom),
        "input_points": prepared.input_points,
        "rendered_points": prepared.rendered_points,
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

    log(f"[OK] Point-cloud view: {image_path}")
    return PointCloudViewResult(
        point_cloud_path=prepared.point_cloud_path,
        image_path=str(image_path),
        metadata_path=str(metadata_path),
        title=title,
        render_style=render_style,
        input_points=prepared.input_points,
        rendered_points=prepared.rendered_points,
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        yaw_deg=float(yaw_deg),
        pitch_deg=float(pitch_deg),
        pan_x=float(pan_x),
        pan_y=float(pan_y),
        zoom=float(zoom),
    )


def render_point_cloud_view_set(
    *,
    point_cloud_path: str | None = None,
    render_data: PointCloudRenderData | None = None,
    output_dir: str,
    views: Sequence[tuple[str, float, float]] = DEFAULT_POINT_CLOUD_API_VIEWS,
    render_style: str = "points",
    width: int = 320,
    height: int = 320,
    max_points: int = 45000,
    log: LogFn = print,
) -> PointCloudViewSetResult:
    prepared = _resolve_render_data(point_cloud_path=point_cloud_path, render_data=render_data, max_points=max_points)
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    image_paths: list[str] = []
    view_names: list[str] = []
    for view_name, yaw_deg, pitch_deg in views:
        view_result = render_point_cloud_view(
            render_data=prepared,
            output_path=str(output_root / f"{_slugify_view_name(view_name)}.png"),
            title=view_name,
            render_style=render_style,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            width=width,
            height=height,
            log=log,
        )
        image_paths.append(view_result.image_path)
        view_names.append(view_name)

    metadata_path = output_root / "view_set.json"
    metadata = {
        "point_cloud_path": prepared.point_cloud_path,
        "output_dir": str(output_root),
        "render_style": render_style,
        "input_points": prepared.input_points,
        "rendered_points": prepared.rendered_points,
        "views": [
            {"name": view_name, "yaw_deg": float(yaw_deg), "pitch_deg": float(pitch_deg), "image_path": image_path}
            for (view_name, yaw_deg, pitch_deg), image_path in zip(views, image_paths, strict=False)
        ],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

    log(f"[OK] Point-cloud API view set: {output_root}")
    return PointCloudViewSetResult(
        point_cloud_path=prepared.point_cloud_path,
        output_dir=str(output_root),
        metadata_path=str(metadata_path),
        render_style=render_style,
        input_points=prepared.input_points,
        rendered_points=prepared.rendered_points,
        view_names=tuple(view_names),
        image_paths=tuple(image_paths),
    )


def render_point_cloud_snapshot(
    *,
    point_cloud_path: str | None = None,
    render_data: PointCloudRenderData | None = None,
    output_path: str | None = None,
    views: Sequence[tuple[str, float, float]] = DEFAULT_POINT_CLOUD_API_VIEWS,
    render_style: str = "points",
    panel_width: int = 300,
    panel_height: int = 300,
    max_points: int = 45000,
    log: LogFn = print,
) -> PointCloudSnapshotResult:
    prepared = _resolve_render_data(point_cloud_path=point_cloud_path, render_data=render_data, max_points=max_points)
    point_cloud = Path(prepared.point_cloud_path)
    image_path = (
        Path(output_path).expanduser().resolve()
        if output_path
        else point_cloud.with_name("point_cloud_snapshot.png")
    )
    image_path.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        _render_view(
            _rotate_points(prepared.aligned_points, yaw_deg=yaw_deg, pitch_deg=pitch_deg),
            prepared.colors,
            title=view_name,
            width=panel_width,
            height=panel_height,
            render_style=render_style,
        )
        for view_name, yaw_deg, pitch_deg in views
    ]
    montage = _compose_panel_grid(panels, columns=2 if len(panels) > 1 else 1, gap=12)
    footer = np.full((84, montage.shape[1], 3), 255, dtype=np.uint8)
    cv2.putText(
        footer,
        f"points: {prepared.input_points}  rendered: {prepared.rendered_points}",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (24, 24, 24),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        footer,
        f"source: {point_cloud.name}",
        (18, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )
    image = np.concatenate([montage, footer], axis=0)
    if not cv2.imwrite(str(image_path), image):
        raise RuntimeError(f"Failed to write point-cloud snapshot: {image_path}")

    metadata_path = image_path.with_suffix(".json")
    metadata = {
        "point_cloud_path": prepared.point_cloud_path,
        "image_path": str(image_path),
        "render_style": render_style,
        "input_points": prepared.input_points,
        "rendered_points": prepared.rendered_points,
        "view_names": [view_name for view_name, _, _ in views],
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "panel_width": int(panel_width),
        "panel_height": int(panel_height),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

    log(f"[OK] Point-cloud snapshot: {image_path}")
    return PointCloudSnapshotResult(
        point_cloud_path=prepared.point_cloud_path,
        image_path=str(image_path),
        metadata_path=str(metadata_path),
        render_style=render_style,
        input_points=prepared.input_points,
        rendered_points=prepared.rendered_points,
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        view_names=tuple(view_name for view_name, _, _ in views),
    )


def snapshot_result_to_dict(result: PointCloudSnapshotResult) -> dict[str, object]:
    return asdict(result)
