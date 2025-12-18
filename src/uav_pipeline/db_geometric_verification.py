from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

try:
    import cv2  # type: ignore
except Exception as e:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR = e
else:  # pragma: no cover
    _CV2_IMPORT_ERROR = None


LogFn = Callable[[str], None]


MAX_IMAGE_ID = 2147483647


@dataclass(frozen=True)
class GeoVerifyConfig:
    ransac_threshold_px: float = 4.0
    confidence: float = 0.9999
    min_inliers: int = 15
    min_inlier_ratio: float = 0.25


def _blob_to_array(blob: bytes, dtype: np.dtype, shape: tuple[int, int]) -> np.ndarray:
    arr = np.frombuffer(blob, dtype=dtype)
    return arr.reshape(shape)


def _array_to_blob(arr: np.ndarray) -> bytes:
    return np.asarray(arr).tobytes()


def _pair_id_to_image_ids(pair_id: int) -> tuple[int, int]:
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) // MAX_IMAGE_ID
    return int(image_id1), int(image_id2)


def _iter_matches(conn: sqlite3.Connection) -> Iterable[tuple[int, np.ndarray]]:
    """
    Yield (pair_id, matches_uint32[N,2]) from COLMAP `matches` table.
    """
    cur = conn.cursor()
    for pair_id, rows, cols, data in cur.execute("SELECT pair_id, rows, cols, data FROM matches"):
        if rows is None or cols is None or data is None:
            continue
        matches = _blob_to_array(data, np.uint32, (int(rows), int(cols)))
        yield int(pair_id), matches


def _get_keypoints(conn: sqlite3.Connection, image_id: int) -> np.ndarray:
    cur = conn.cursor()
    row = cur.execute("SELECT rows, cols, data FROM keypoints WHERE image_id=?", (int(image_id),)).fetchone()
    if row is None:
        raise RuntimeError(f"database.db 缺少 image_id={image_id} 的 keypoints")
    rows, cols, data = row
    kps = _blob_to_array(data, np.float32, (int(rows), int(cols)))
    if kps.shape[1] >= 2:
        return kps[:, :2]
    raise RuntimeError(f"image_id={image_id} 的 keypoints shape 异常: {kps.shape}")


def geometric_verification_db(
    database_path: str,
    *,
    cfg: GeoVerifyConfig | None = None,
    log: LogFn | None = None,
) -> None:
    """
    Perform geometric verification in Python and write `two_view_geometries` for COLMAP mapper.

    This is a compatibility fallback for COLMAP builds that don't ship a standalone
    `geometric_verification` command.
    """
    if cv2 is None:  # pragma: no cover
        raise RuntimeError(
            "无法导入 opencv-python（cv2），无法做几何验证。\n"
            f"原始错误: {_CV2_IMPORT_ERROR}"
        )

    cfg = cfg or GeoVerifyConfig()
    log = log or (lambda _m: None)

    conn = sqlite3.connect(database_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")

    # Start fresh.
    conn.execute("DELETE FROM two_view_geometries")
    conn.commit()

    keypoints_cache: dict[int, np.ndarray] = {}

    def get_kps(img_id: int) -> np.ndarray:
        if img_id not in keypoints_cache:
            keypoints_cache[img_id] = _get_keypoints(conn, img_id)
        return keypoints_cache[img_id]

    inserted = 0
    skipped = 0
    for pair_id, matches in _iter_matches(conn):
        if matches.size == 0:
            skipped += 1
            continue

        img1, img2 = _pair_id_to_image_ids(pair_id)
        kps1 = get_kps(img1)
        kps2 = get_kps(img2)

        if matches.shape[0] < 8:
            skipped += 1
            continue

        pts1 = kps1[matches[:, 0].astype(np.int64)]
        pts2 = kps2[matches[:, 1].astype(np.int64)]

        method = cv2.FM_RANSAC
        # Prefer USAC_MAGSAC if available (OpenCV >= 4.5+).
        if hasattr(cv2, "USAC_MAGSAC"):
            method = cv2.USAC_MAGSAC

        F, mask = cv2.findFundamentalMat(
            pts1,
            pts2,
            method,
            cfg.ransac_threshold_px,
            cfg.confidence,
        )
        if mask is None:
            skipped += 1
            continue

        inlier_mask = mask.ravel().astype(bool)
        inliers = matches[inlier_mask]
        if inliers.shape[0] < cfg.min_inliers:
            skipped += 1
            continue
        if inliers.shape[0] / max(1, matches.shape[0]) < cfg.min_inlier_ratio:
            skipped += 1
            continue

        if F is None or np.asarray(F).shape != (3, 3):
            F = np.eye(3, dtype=np.float64)
        F = np.asarray(F, dtype=np.float64)

        # Fill remaining fields with defaults.
        E = np.eye(3, dtype=np.float64)
        H = np.eye(3, dtype=np.float64)
        qvec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        tvec = np.zeros(3, dtype=np.float64)
        config_val = 2  # matches deep_image_matching default

        conn.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                int(pair_id),
                int(inliers.shape[0]),
                2,
                _array_to_blob(inliers.astype(np.uint32)),
                int(config_val),
                _array_to_blob(F),
                _array_to_blob(E),
                _array_to_blob(H),
                _array_to_blob(qvec),
                _array_to_blob(tvec),
            ),
        )
        inserted += 1
        if inserted % 500 == 0:
            log(f"[INFO] geometric verification: inserted={inserted}, skipped={skipped}")

    conn.commit()
    conn.close()
    log(f"[OK] geometric verification done: inserted={inserted}, skipped={skipped}")

