"""CLI for Proposal-grounded Association over frozen grounding observations."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .backend import (
    load_grounding_observations,
    load_proposals,
    load_scene,
    project_points,
    scale_point,
    scene_parser,
    visit_to_videos,
)
from .config import load_config, path_value
from .pga import eligible_proposals, local_surface_support


def point_to_proposal_distances(
    projection: np.ndarray,
    union: np.ndarray,
    point_to_proposals: dict[int, list[int]],
    number_of_proposals: int,
    point_x: float,
    point_y: float,
) -> np.ndarray:
    visible = projection[:, 2] == 1
    distances = np.full(number_of_proposals, np.inf, dtype=np.float64)
    if not visible.any():
        return distances
    yy = projection[visible, 0].astype(np.float64)
    xx = projection[visible, 1].astype(np.float64)
    pixel_distances = np.hypot(xx - point_x, yy - point_y)
    local_indices = np.flatnonzero(visible)
    for local_index, distance in zip(local_indices.tolist(), pixel_distances.tolist()):
        global_index = int(union[local_index])
        for proposal in point_to_proposals.get(global_index, ()):
            distances[proposal] = min(distances[proposal], distance)
    return distances


def associate_query(
    query: dict,
    *,
    fun3du_root: Path,
    parser,
    xyz: np.ndarray,
    proposals: list[np.ndarray],
    video_data: dict,
    grounding_dir: Path,
    device: str,
    max_point_distance_px: float,
    surface_radius_px: float,
    projection_cache: dict,
) -> dict:
    visit, description_id = str(query["visit"]), str(query["desc_id"])
    metadata = load_grounding_observations(
        grounding_dir / f"{visit}_{description_id}.npz"
    )
    union_values: list[int] = []
    point_to_proposals: dict[int, list[int]] = defaultdict(list)
    for proposal_id, indices in enumerate(proposals):
        for index in indices.tolist():
            union_values.append(int(index))
            point_to_proposals[int(index)].append(proposal_id)
    union = (
        np.unique(np.asarray(union_values, dtype=np.int64))
        if union_values else np.zeros(0, dtype=np.int64)
    )
    union_position = {int(index): i for i, index in enumerate(union.tolist())}
    proposal_locations = [
        np.asarray([union_position[int(index)] for index in proposal
                    if int(index) in union_position], dtype=np.int64)
        for proposal in proposals
    ]
    observations = []
    if metadata is not None and proposals:
        points = np.asarray(metadata["points"])
        original_dimensions = metadata["original_dimensions"]
        for index in range(int(metadata["frame_ids"].shape[0])):
            point = np.asarray(points[index] if points.ndim == 2 else points).reshape(-1)
            if point.size < 2 or not np.isfinite(point[:2]).all() or not point[:2].any():
                continue
            video_id = str(metadata["video_ids"][index])
            frame_id = str(metadata["frame_ids"][index])
            frame_key = (video_id, frame_id)
            if frame_key not in projection_cache:
                projection_cache[frame_key] = project_points(
                    fun3du_root, parser, video_data, xyz[union], video_id, frame_id, device
                )
            packed = projection_cache[frame_key]
            if packed is None:
                continue
            projection, (height, width) = packed
            original_hw = (
                original_dimensions[index]
                if original_dimensions is not None
                and original_dimensions.ndim == 2
                and index < len(original_dimensions)
                else None
            )
            point_x, point_y = scale_point(point[:2], original_hw, width, height)
            distances = point_to_proposal_distances(
                projection, union, point_to_proposals, len(proposals), point_x, point_y
            )
            order = eligible_proposals(distances, max_point_distance_px)
            if order.size == 0:
                continue
            support = [
                local_surface_support(
                    projection, proposal_locations[int(proposal)],
                    point_x, point_y, surface_radius_px
                )
                for proposal in order
            ]
            nearest = int(order[0])
            record = {
                "video_id": video_id,
                "frame_id": frame_id,
                "grounding_point": [float(point[0]), float(point[1])],
                "nearest_proposal": nearest,
                "nearest_distance": float(distances[nearest]),
                "nearest_support": float(support[0]),
                "second_proposal": -1,
                "second_distance": None,
                "second_support": None,
            }
            if order.size >= 2:
                second = int(order[1])
                record.update({
                    "second_proposal": second,
                    "second_distance": float(distances[second]),
                    "second_support": float(support[1]),
                })
            observations.append(record)
    return {
        "visit": visit,
        "desc_id": description_id,
        "instruction": str(query.get("query", "")),
        "number_of_proposals": len(proposals),
        "number_of_grounding_frames": 0 if metadata is None else int(len(metadata["frame_ids"])),
        "number_of_valid_observations": len(observations),
        "observations": observations,
    }


def run(config: dict, visits: set[str] | None = None,
        description_prefixes: set[str] | None = None) -> Path:
    fun3du_root = path_value(config, "fun3du_root")
    dataset_root = path_value(config, "dataset_root")
    bank_root = path_value(config, "proposal_bank")
    work_dir = path_value(config, "work_dir")
    split = str(config.get("split", "val"))
    contracts_path = Path(config.get("contracts", work_dir / "contracts.json"))
    grounding_dir = Path(config.get("grounding_observations", work_dir / "grounding/frames"))
    output = Path(config.get("pga_output", work_dir / "pga_associations.json"))
    contracts_payload = json.loads(contracts_path.read_text())
    queries = contracts_payload.get("queries", contracts_payload)
    if visits:
        queries = [row for row in queries if str(row["visit"]) in visits]
    if description_prefixes:
        queries = [row for row in queries if any(
            str(row["desc_id"]).startswith(prefix) for prefix in description_prefixes
        )]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for query in queries:
        grouped[str(query["visit"])].append(query)
    parser = scene_parser(fun3du_root, dataset_root, split)
    visits2videos = visit_to_videos(fun3du_root, dataset_root, split)
    rows = []
    for visit, visit_queries in sorted(grouped.items()):
        xyz, _ = load_scene(parser, visit)
        proposals = load_proposals(bank_root, visit, len(xyz))
        video_data = {
            video: {
                "depth_paths": parser.get_depth_frames(visit, video),
                "intrinsics": parser.get_camera_intrinsics(visit, video),
                "poses": parser.get_camera_trajectory(visit, video),
            }
            for video in visits2videos.get(visit, [])
        }
        projection_cache = {}
        for index, query in enumerate(visit_queries, 1):
            row = associate_query(
                query,
                fun3du_root=fun3du_root,
                parser=parser,
                xyz=xyz,
                proposals=proposals,
                video_data=video_data,
                grounding_dir=grounding_dir,
                device=str(config.get("device", "cuda:0")),
                max_point_distance_px=float(config["pga"]["max_point_distance_px"]),
                surface_radius_px=float(config["pga"]["surface_radius_px"]),
                projection_cache=projection_cache,
            )
            rows.append(row)
            print(f"[PGA {visit}] {index}/{len(visit_queries)} "
                  f"{row['desc_id'][:8]} valid={row['number_of_valid_observations']}", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema_version": 1,
        "method": "Proposal-grounded Association",
        "number_of_queries": len(rows),
        "queries": rows,
    }, indent=2, ensure_ascii=False))
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--visits", default="")
    parser.add_argument("--descriptions", default="")
    args = parser.parse_args()
    output = run(
        load_config(args.config),
        {value for value in args.visits.split(",") if value} or None,
        {value for value in args.descriptions.split(",") if value} or None,
    )
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()

