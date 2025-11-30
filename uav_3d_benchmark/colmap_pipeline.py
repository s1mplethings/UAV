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


def _read_camera_from_sparse(sparse_known_dir: str):
    cam_path = os.path.join(sparse_known_dir, "cameras.txt")
    with open(cam_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            model = parts[1]
            params = parts[4:]
            return model, params
    return None, None


def _sparse_exists(sparse_dir: str) -> bool:
    return os.path.exists(os.path.join(sparse_dir, "cameras.bin")) or os.path.exists(
        os.path.join(sparse_dir, "cameras.txt")
    )


def _dense_fused_exists(dense_dir: str) -> bool:
    return os.path.exists(os.path.join(dense_dir, "fused.ply"))


def _undistorted_exists(dense_dir: str) -> bool:
    return os.path.isdir(os.path.join(dense_dir, "images"))


def run_colmap_pipeline(
    image_path: str,
    sparse_known_dir: str,
    work_root: str,
    max_image_size: int = 1000,
    use_gpu: bool = True,
    run_dense: bool = True,
    resume: bool = True,
):
    """
    image_path: 原始图片所在目录
    sparse_known_dir: 已知位姿的 cameras.txt / images.txt / points3D.txt 目录
    work_root: 本次实验输出目录
    max_image_size: 特征提取前的最大边长下采样
    use_gpu: PatchMatch 尝试启用 GPU（其他步骤不传 GPU 选项以兼容旧版 COLMAP）
    run_dense: 是否运行稠密部分
    resume: 若输出已存在则跳过对应步骤，避免从头跑
    """
    os.makedirs(work_root, exist_ok=True)
    db_path = os.path.join(work_root, "database.db")
    cam_model, cam_params = _read_camera_from_sparse(sparse_known_dir)
    image_reader_opts = []
    if cam_model and cam_params:
        image_reader_opts = [
            "--ImageReader.camera_model",
            cam_model,
            "--ImageReader.single_camera",
            "1",
            "--ImageReader.camera_params",
            ",".join(cam_params),
        ]

    # 1) 特征提取
    if not (resume and os.path.exists(db_path)):
        run_cmd(
            [
                COLMAP_BIN,
                "feature_extractor",
                "--database_path",
                db_path,
                "--image_path",
                image_path,
                "--SiftExtraction.max_image_size",
                str(max_image_size),
            ]
            + image_reader_opts
        )
        # 2) 匹配（顺序）
        run_cmd(
            [
                COLMAP_BIN,
                "sequential_matcher",
                "--database_path",
                db_path,
            ]
        )

    # 3) 已知位姿三角化
    sparse_out = os.path.join(work_root, "sparse")
    if os.path.exists(sparse_out) and not os.path.isdir(sparse_out):
        os.remove(sparse_out)
    os.makedirs(sparse_out, exist_ok=True)
    if not (resume and _sparse_exists(sparse_out)):
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

    if not run_dense:
        return None

    # 4) 稠密部分
    dense_root = os.path.join(work_root, "dense")
    if os.path.exists(dense_root) and not os.path.isdir(dense_root):
        os.remove(dense_root)
    os.makedirs(dense_root, exist_ok=True)
    if not (resume and _undistorted_exists(dense_root)):
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
    if not (resume and _dense_fused_exists(dense_root)):
        run_cmd(
            [
                COLMAP_BIN,
                "patch_match_stereo",
                "--workspace_path",
                dense_root,
                "--PatchMatchStereo.gpu_index",
                "0" if use_gpu else "-1",
                "--PatchMatchStereo.num_samples",
                "10",
                "--PatchMatchStereo.num_iterations",
                "3",
                "--PatchMatchStereo.geom_consistency",
                "0",
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

    fused_path = os.path.join(dense_root, "fused.ply")
    return fused_path
