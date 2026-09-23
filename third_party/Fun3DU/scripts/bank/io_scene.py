"""Scene IO helpers: cropped laser + GT annot masks (full→crop index map)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d

from utils.sun3d.data_parser import DataParser

DEFAULT_DATA_ROOT = str(Path(__file__).resolve().parents[4] / "data" / "scenefun3d")


def make_parser(split: str = "val", data_root: str = DEFAULT_DATA_ROOT) -> DataParser:
    return DataParser(data_root, split)


def load_cropped_xyz(
    parser: DataParser, visit: str
) -> Tuple[np.ndarray, np.ndarray, o3d.geometry.PointCloud]:
    pcd = parser.get_cropped_laser_scan(visit, parser.get_laser_scan(visit))
    xyz = np.asarray(pcd.points, dtype=np.float64)
    rgb = np.asarray(pcd.colors, dtype=np.float64)
    if rgb.size == 0:
        rgb = np.full((len(xyz), 3), 0.55, dtype=np.float64)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    return xyz, rgb, pcd


def load_gt_annots(
    parser: DataParser, visit: str, skip_exclude: bool = True
) -> List[Dict]:
    """Return GT masks on *cropped* laser (same length as load_cropped_xyz).

    Annotation indices in JSON are on the uncropped scan; we apply crop_mask
    exactly like DataParser.get_grouped_annotation.
    """
    crop_mask = parser.get_crop_mask(visit)  # full-length 0/1
    n_full = int(crop_mask.shape[0])
    n_crop = int((crop_mask == 1).sum())

    raw = parser.get_annotations(visit, group_excluded_points=True)
    out: List[Dict] = []
    for ann in raw:
        label = str(ann.get("label", ""))
        if skip_exclude and label == "exclude":
            continue
        idxs = np.asarray(ann.get("indices") or [], dtype=np.int64)
        full = np.zeros(n_full, dtype=bool)
        if idxs.size:
            valid = (idxs >= 0) & (idxs < n_full)
            full[idxs[valid]] = True
        cropped = full[crop_mask == 1]
        if cropped.shape[0] != n_crop:
            raise RuntimeError(
                f"crop mismatch: cropped={cropped.shape[0]} expected={n_crop}"
            )
        out.append(
            {
                "annot_id": str(ann["annot_id"]),
                "label": label,
                "gt": cropped,
                "gt_n": int(cropped.sum()),
            }
        )
    return out


def aabb_of_mask(xyz: np.ndarray, mask: np.ndarray) -> Dict[str, List[float]]:
    pts = xyz[mask]
    if len(pts) == 0:
        return {"min": [0.0, 0.0, 0.0], "max": [0.0, 0.0, 0.0]}
    return {
        "min": pts.min(axis=0).astype(float).tolist(),
        "max": pts.max(axis=0).astype(float).tolist(),
    }


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb, 0, 1).astype(np.float64))
    path.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(path), pcd)


def save_bool_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), mask.astype(np.bool_))


def load_bool_mask(path: Path) -> np.ndarray:
    return np.load(str(path)).astype(bool)
