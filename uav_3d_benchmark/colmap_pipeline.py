import os
import subprocess


def resolve_colmap_bin():
    env_bin = os.environ.get("COLMAP_BIN")
    candidates = [
        env_bin,
        r"C:\Program Files\COLMAP\bin\colmap.exe",
        r"C:\Program Files\COLMAP\colmap.bat",
        "colmap",
    ]
    for c in candidates:
        if not c:
            continue
        if os.path.exists(c) or c == "colmap":
            return c
    return "colmap"


COLMAP_BIN = resolve_colmap_bin()


def run_cmd(cmd, cwd=None):
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd)


def run_colmap_pipeline(
    image_path: str,
    sparse_known_dir: str,
    work_root: str,
):
    """
    image_path: 原始图片所在目录
    sparse_known_dir: 已知位姿的 cameras.txt / images.txt / points3D.txt 目录
    work_root: 本次实验输出目录
    """
    os.makedirs(work_root, exist_ok=True)
    db_path = os.path.join(work_root, "database.db")
    run_cmd(
        [
            COLMAP_BIN,
            "feature_extractor",
            "--database_path",
            db_path,
            "--image_path",
            image_path,
        ]
    )
    run_cmd(
        [
            COLMAP_BIN,
            "exhaustive_matcher",
            "--database_path",
            db_path,
        ]
    )
    sparse_out = os.path.join(work_root, "sparse")
    run_cmd(
        [
            COLMAP_BIN,
            "point_triangulator",
            "--database_path",
            db_path,
            "--image_path",
            image_path,
            "--input_path",
            sparse_known_dir,
            "--output_path",
            sparse_out,
        ]
    )
    dense_root = os.path.join(work_root, "dense")
    run_cmd(
        [
            COLMAP_BIN,
            "image_undistorter",
            "--image_path",
            image_path,
            "--input_path",
            sparse_out,
            "--output_path",
            dense_root,
        ]
    )
    run_cmd(
        [
            COLMAP_BIN,
            "patch_match_stereo",
            "--workspace_path",
            dense_root,
        ]
    )
    fused_path = os.path.join(dense_root, "fused.ply")
    run_cmd(
        [
            COLMAP_BIN,
            "stereo_fusion",
            "--workspace_path",
            dense_root,
            "--output_path",
            fused_path,
        ]
    )
    return fused_path
