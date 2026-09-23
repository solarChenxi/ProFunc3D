"""Thin adapter to the frozen Fun3DU data/model backend.

ProFunc3D changes association and aggregation, while reusing the same frozen
instruction parser, frame retriever, 2D grounder and SceneFun3D calibration
code as the reproduced Fun3DU baseline.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch


def activate_fun3du(root: str | Path) -> Path:
    root = Path(root).expanduser().resolve()
    if not (root / "utils/sun3d/data_parser.py").is_file():
        raise FileNotFoundError(f"Fun3DU backend not found at {root}")
    for entry in (root, root / "scripts", root / "scripts/bank"):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    return root


def scene_parser(fun3du_root: Path, dataset_root: Path, split: str):
    activate_fun3du(fun3du_root)
    from utils.sun3d.data_parser import DataParser

    return DataParser(str(dataset_root), split)


def visit_to_videos(fun3du_root: Path, dataset_root: Path, split: str) -> dict:
    activate_fun3du(fun3du_root)
    from utils import io as fun3du_io

    return fun3du_io.get_visit_to_videos(str(dataset_root), split)


def load_scene(parser, visit: str) -> tuple[np.ndarray, np.ndarray]:
    point_cloud = parser.get_cropped_laser_scan(visit, parser.get_laser_scan(visit))
    xyz = np.asarray(point_cloud.points, dtype=np.float64)
    rgb = np.asarray(point_cloud.colors, dtype=np.float64)
    if rgb.size == 0:
        rgb = np.full((len(xyz), 3), 0.55, dtype=np.float64)
    if rgb.max(initial=0.0) > 1.5:
        rgb /= 255.0
    return xyz, rgb


def load_grounding_observations(path: Path) -> dict | None:
    """Read sparse point metadata without decompressing SAM masks."""
    if not path.is_file():
        return None
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if "frame_ids.npy" not in names:
            return None

        def array(name: str) -> np.ndarray:
            return np.load(io.BytesIO(archive.read(name)))

        frame_ids = array("frame_ids.npy")
        if frame_ids.shape[0] == 1 and str(frame_ids[0]) == "0":
            return None
        return {
            "frame_ids": frame_ids,
            "video_ids": array("video_ids.npy"),
            "points": array("points.npy"),
            "original_dimensions": (
                array("orig_dims.npy") if "orig_dims.npy" in names else None
            ),
        }


def scale_point(
    point: np.ndarray, original_hw: np.ndarray | None, width: int, height: int
) -> tuple[float, float]:
    x, y = float(point[0]), float(point[1])
    if original_hw is None or original_hw.size < 2:
        return x, y
    original_height, original_width = float(original_hw[0]), float(original_hw[1])
    if original_height <= 1 or original_width <= 1:
        return x, y
    return x * width / original_width, y * height / original_height


def lookup_timestamp(mapping: dict, frame_id: str):
    if frame_id in mapping:
        return mapping[frame_id]
    try:
        target = float(frame_id)
    except ValueError:
        return None
    candidates = []
    for key, value in mapping.items():
        try:
            candidates.append((abs(float(key) - target), value))
        except (TypeError, ValueError):
            pass
    if not candidates:
        return None
    distance, value = min(candidates, key=lambda row: row[0])
    return value if distance < 0.05 else None


def project_points(
    fun3du_root: Path,
    parser,
    video_data: dict,
    xyz: np.ndarray,
    video_id: str,
    frame_id: str,
    device: str,
) -> tuple[np.ndarray, tuple[int, int]] | None:
    """Project 3D points into one calibrated RGB-D observation."""
    activate_fun3du(fun3du_root)
    from utils.sun3d.fusion_util import PointCloudToImageMapper

    data = video_data.get(video_id)
    if data is None:
        return None
    depth_path = lookup_timestamp(data["depth_paths"], frame_id)
    intrinsic_path = lookup_timestamp(data["intrinsics"], frame_id)
    if depth_path is None or intrinsic_path is None:
        return None
    try:
        depth = parser.read_depth_frame(depth_path)
        pose = parser.get_nearest_pose(frame_id, data["poses"])
        intrinsic = parser.read_camera_intrinsics(intrinsic_path, format="matrix")
    except Exception:
        return None
    if pose is None or len(xyz) == 0:
        return None
    height, width = depth.shape
    mapper = PointCloudToImageMapper((width, height))
    coordinates = torch.as_tensor(xyz, dtype=torch.float64).to(device)
    projection = mapper.compute_multi_masked_mapping(
        pose,
        coordinates,
        np.ones((1, height, width), dtype=np.uint8),
        depth,
        intrinsic,
        device,
    )[0]
    return projection, (height, width)


def load_proposals(bank_root: Path, visit: str, number_of_points: int) -> list[np.ndarray]:
    archive = np.load(bank_root / "merged" / f"{visit}.npz", allow_pickle=False)
    concatenated = archive["idx_concat"].astype(np.int64)
    offsets = archive["idx_offsets"].astype(np.int64)
    proposals = []
    for index in range(len(offsets) - 1):
        values = concatenated[offsets[index]:offsets[index + 1]]
        proposals.append(values[(values >= 0) & (values < number_of_points)])
    return proposals


def write_ascii_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors = np.clip(np.rint(rgb * 255), 0, 255).astype(np.uint8)
    lines = [
        "ply", "format ascii 1.0", f"element vertex {len(xyz)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue", "end_header",
    ]
    with path.open("w") as stream:
        stream.write("\n".join(lines) + "\n")
        for point, color in zip(xyz, colors):
            stream.write(
                f"{point[0]:.7g} {point[1]:.7g} {point[2]:.7g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )

