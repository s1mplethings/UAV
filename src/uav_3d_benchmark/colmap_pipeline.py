import os
import shutil
import subprocess
from pathlib import Path

# 默认 COLMAP 路径，可用环境变量 COLMAP_BIN 覆盖
DEFAULT_COLMAP_BIN = os.environ.get(
    "COLMAP_BIN",
    r"C:\Program Files\COLMAP\bin\colmap.exe",
)


def run_cmd(cmd, cwd=None):
    """打印并执行命令，失败抛异常。"""
    print(">>", " ".join(str(c) for c in cmd))
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd is not None else None)


def _read_camera_model_from_sparse(sparse_dir: Path) -> str:
    """
    从 sparse_known/cameras.txt 读取模型名称（PINHOLE / SIMPLE_RADIAL 等）。
    若缺失则返回 SIMPLE_RADIAL 作为兜底。
    """
    cam_file = sparse_dir / "cameras.txt"
    if not cam_file.exists():
        return "SIMPLE_RADIAL"

    with cam_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                return parts[1]
    return "SIMPLE_RADIAL"


def _write_sequential_pairs(images_dir: Path, pairs_path: Path, neighbors: int = 10):
    """
    写顺序 pairs.txt，确保 PatchMatch 有邻接视图生成深度。
    """
    imgs = sorted([p.name for p in images_dir.iterdir() if p.is_file()])
    pairs_path.parent.mkdir(parents=True, exist_ok=True)
    with pairs_path.open("w", encoding="utf-8") as f:
        for i, name in enumerate(imgs):
            neigh = []
            for j in range(max(0, i - neighbors), min(len(imgs), i + neighbors + 1)):
                if j == i:
                    continue
                neigh.append(imgs[j])
            if neigh:
                f.write(name + " " + " ".join(neigh) + "\n")


def run_colmap_pipeline(image_folder, sparse_known, work_root):
    """
    已知位姿的精简 COLMAP 流程（跳过 point_triangulator）：
      1) feature_extractor（模型与 sparse_known 一致，单相机）
      2) sequential_matcher
      3) image_undistorter（直接用 sparse_known）
      4) patch_match_stereo（写 pairs.txt，设定深度范围）
      5) stereo_fusion
    sparse_known 需至少包含 cameras.txt / images.txt（points3D 可为空）。
    """
    image_folder = Path(image_folder).resolve()
    sparse_known = Path(sparse_known).resolve()
    work_root = Path(work_root).resolve()

    work_root.mkdir(parents=True, exist_ok=True)

    database_path = work_root / "database.db"
    dense_root = work_root / "dense"
    dense_root.mkdir(parents=True, exist_ok=True)

    # 清理旧结果
    work_root.mkdir(parents=True, exist_ok=True)
    if database_path.exists():
        database_path.unlink()
    if dense_root.exists():
        shutil.rmtree(dense_root)
    dense_root.mkdir(parents=True, exist_ok=True)

    colmap_bin = DEFAULT_COLMAP_BIN

    # 读取 sparse_known 的相机模型，确保与 feature_extractor 一致
    camera_model = _read_camera_model_from_sparse(sparse_known)
    print(f"[INFO] Using camera model from sparse_known: {camera_model}")

    # 1) 特征提取
    run_cmd(
        [
            colmap_bin,
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_folder),
            "--ImageReader.camera_model",
            camera_model,
            "--ImageReader.single_camera",
            "1",
        ]
    )

    # 2) 顺序匹配
    run_cmd(
        [
            colmap_bin,
            "sequential_matcher",
            "--database_path",
            str(database_path),
        ]
    )

    # 3) 直接使用已知 sparse（跳过 point_triangulator）
    sparse = sparse_known

    # 4) 稠密工作区
    run_cmd(
        [
            colmap_bin,
            "image_undistorter",
            "--image_path",
            str(image_folder),
            "--input_path",
            str(sparse),
            "--output_path",
            str(dense_root),
            "--output_type",
            "COLMAP",
        ]
    )

    # 写 pairs.txt 供 PatchMatch 使用
    _write_sequential_pairs(dense_root / "images", dense_root / "stereo" / "pairs.txt")

    # 5) PatchMatch Stereo；使用几何一致性并设定深度范围
    run_cmd(
        [
            colmap_bin,
            "patch_match_stereo",
            "--workspace_path",
            str(dense_root),
            "--workspace_format",
            "COLMAP",
            "--PatchMatchStereo.geom_consistency",
            "true",
            "--PatchMatchStereo.depth_min",
            "0.2",
            "--PatchMatchStereo.depth_max",
            "20.0",
            "--PatchMatchStereo.filter",
            "1",
        ]
    )

    # 6) 融合
    fused_path = dense_root / "fused.ply"
    run_cmd(
        [
            colmap_bin,
            "stereo_fusion",
            "--workspace_path",
            str(dense_root),
            "--workspace_format",
            "COLMAP",
            "--input_type",
            "geometric",
            "--output_path",
            str(fused_path),
        ]
    )

    print(f"[INFO] Fused point cloud saved to: {fused_path}")
    return fused_path
