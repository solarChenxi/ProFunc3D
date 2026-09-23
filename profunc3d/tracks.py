"""Build relation-consistent physical parent-instance tracks for IAR."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import DBSCAN

from .backend import (
    load_grounding_observations,
    load_scene,
    project_points,
    scene_parser,
    visit_to_videos,
)
from .config import load_config, path_value
from .iar import frame_completeness, observation_quality


def sample_timeline(items: list[tuple[str, str]], minimum_gap_s: float) -> list[tuple[str, str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for video_id, frame_id in items:
        grouped[str(video_id)].append(str(frame_id))
    selected = []
    for video_id, frame_ids in grouped.items():
        previous = -1e30
        for frame_id in sorted(set(frame_ids), key=float):
            if float(frame_id) - previous >= minimum_gap_s:
                selected.append((video_id, frame_id))
                previous = float(frame_id)
    return sorted(selected, key=lambda row: (row[0], float(row[1])))


def lift_mask(projection: np.ndarray, sampled_indices: np.ndarray, mask: np.ndarray) -> np.ndarray:
    visible = projection[:, 2] == 1
    local = np.flatnonzero(visible)
    if local.size == 0:
        return np.zeros(0, dtype=np.int64)
    height, width = mask.shape
    yy = np.clip(np.rint(projection[visible, 0]).astype(int), 0, height - 1)
    xx = np.clip(np.rint(projection[visible, 1]).astype(int), 0, width - 1)
    return sampled_indices[local[mask[yy, xx] > 0]]


def cluster_parent_observations(observations: list[dict], radius_m: float) -> np.ndarray:
    if not observations:
        return np.zeros(0, dtype=int)
    labels = DBSCAN(eps=radius_m, min_samples=2).fit_predict(
        np.asarray([row["centroid"] for row in observations])
    )
    next_label = int(labels.max() + 1)
    for index, label in enumerate(labels):
        if label < 0:
            labels[index] = next_label
            next_label += 1
        observations[index]["track_id"] = int(labels[index])
    return labels


def scalar_relation(relation: dict, parent: dict, reference: dict) -> float:
    predicate = relation["predicate"]
    dx = float(parent["center_x"] - reference["center_x"])
    dy = float(parent["center_y"] - reference["center_y"])
    distance = math.hypot(dx, dy)
    if predicate == "right_of":
        return float(dx > 0.04)
    if predicate == "left_of":
        return float(dx < -0.04)
    if predicate == "above":
        return float(dy < -0.04)
    if predicate == "below":
        return float(dy > 0.04)
    if predicate in {"near", "next_to"}:
        return float(math.exp(-distance / 0.28))
    if predicate in {"behind", "in_front_of"}:
        return float(math.exp(-distance / 0.22))
    if predicate == "on_top_of":
        parent_above = relation.get("subject") == "parent"
        vertical = dy < -0.02 if parent_above else dy > 0.02
        return float(vertical) * float(math.exp(-abs(dx) / 0.30))
    if predicate == "attached_to":
        return float(math.exp(-distance / 0.20))
    return 0.0


def relation_scores(
    relation: dict,
    parent: dict,
    references: list[dict],
    reference_groups: list[set[str]],
) -> list[float]:
    nearby = [
        row for row in references
        if row["video_id"] == parent["video_id"]
        and abs(float(row["frame_id"]) - float(parent["frame_id"])) < 0.30
    ]
    if relation["predicate"] != "between":
        scores = []
        for labels in reference_groups:
            values = [scalar_relation(relation, parent, row)
                      for row in nearby if row["label"] in labels]
            if values:
                scores.append(max(values))
        return scores if len(scores) == len(reference_groups) else []
    if len(reference_groups) >= 2:
        first = [row for row in nearby if row["label"] in reference_groups[0]]
        second = [row for row in nearby if row["label"] in reference_groups[1]]
        pairs = [(a, b) for a in first for b in second]
    else:
        first = [row for row in nearby if reference_groups and row["label"] in reference_groups[0]]
        pairs = [(first[i], first[j]) for i in range(len(first)) for j in range(i + 1, len(first))]
    values = []
    for first, second in pairs:
        midpoint_x = (first["center_x"] + second["center_x"]) / 2
        midpoint_y = (first["center_y"] + second["center_y"]) / 2
        separation = math.hypot(
            first["center_x"] - second["center_x"],
            first["center_y"] - second["center_y"],
        )
        if separation >= 0.05:
            values.append(math.exp(-math.hypot(
                parent["center_x"] - midpoint_x,
                parent["center_y"] - midpoint_y,
            ) / 0.22))
    return [max(values)] if values else []


def available_keys(keys: list[str], object_index: dict) -> list[str]:
    return [str(key) for key in keys if key in object_index and object_index[key]]


def temporal_ranges(grounding: dict | None, context_s: float) -> dict[str, tuple[float, float]]:
    if grounding is None:
        return {}
    ranges = {}
    for video_id in set(map(str, grounding["video_ids"][:50])):
        values = [float(frame_id) for video, frame_id in zip(
            grounding["video_ids"][:50], grounding["frame_ids"][:50]
        ) if str(video) == video_id]
        if values:
            ranges[video_id] = (min(values) - context_s, max(values) + context_s)
    return ranges


def build_query_track(
    contract: dict,
    *,
    config: dict,
    parser,
    videos: list[str],
    xyz: np.ndarray,
    overrides: dict,
) -> dict:
    visit, description_id = str(contract["visit"]), str(contract["desc_id"])
    mask_type = str(config.get("object_masks", "owl2_rsam_v2"))
    object_index = parser.get_mask_index(visit, mask_type).get("objects", {})
    override = overrides.get(description_id, {})
    parent_keys = override.get("retrieval_keys") or contract["parent"].get("retrieval_keys", [])
    parent_labels = set(available_keys(parent_keys, object_index))
    reference_groups = [
        set(available_keys(reference.get("retrieval_keys", []), object_index))
        for reference in contract.get("anchors", [])
    ]
    if not parent_labels or not reference_groups or not all(reference_groups):
        return {
            "visit": visit, "desc_id": description_id,
            "instruction": contract.get("query", ""),
            "selected_track": None, "selected_track_detections": [],
            "tracks": [], "status": "missing_parent_or_reference_detection",
        }
    grounding = load_grounding_observations(
        Path(config["grounding_observations"]) / f"{visit}_{description_id}.npz"
    )
    ranges = temporal_ranges(grounding, float(config["iar"]["context_window_s"]))
    parent_frames = []
    reference_frames = []
    for label in parent_labels:
        for value in object_index.get(label, []):
            video_id, frame_id = value.split()
            if video_id in ranges and ranges[video_id][0] <= float(frame_id) <= ranges[video_id][1]:
                parent_frames.append((video_id, frame_id))
    all_reference_labels = set().union(*reference_groups)
    for label in all_reference_labels:
        for value in object_index.get(label, []):
            video_id, frame_id = value.split()
            if video_id in ranges and ranges[video_id][0] <= float(frame_id) <= ranges[video_id][1]:
                reference_frames.append((video_id, frame_id))
    gap = float(config["iar"]["timeline_sampling_s"])
    frame_roles = {(video, frame): {"parent"} for video, frame in sample_timeline(parent_frames, gap)}
    for video, frame in sample_timeline(reference_frames, gap):
        frame_roles.setdefault((video, frame), set()).add("reference")
    video_data = {
        video: {
            "depth_paths": parser.get_depth_frames(visit, video),
            "intrinsics": parser.get_camera_intrinsics(visit, video),
            "poses": parser.get_camera_trajectory(visit, video),
        }
        for video in videos
    }
    stride = int(config["iar"]["point_stride"])
    sampled_indices = np.arange(0, len(xyz), stride, dtype=np.int64)
    sampled_xyz = xyz[sampled_indices]
    observations = []
    dataset_root = Path(config["dataset_root"])
    split = str(config.get("split", "val"))
    fun3du_root = Path(config["fun3du_root"])
    for video_id, frame_id in sorted(frame_roles, key=lambda row: (row[0], float(row[1]))):
        path = dataset_root / split / visit / video_id / mask_type / f"{video_id}_{frame_id}.npz"
        if not path.is_file():
            continue
        packed = project_points(
            fun3du_root, parser, video_data, sampled_xyz,
            video_id, frame_id, str(config.get("device", "cuda:0")),
        )
        if packed is None:
            continue
        projection, _ = packed
        archive = np.load(path)
        for index, raw_label in enumerate(archive["labels"].tolist()):
            label = str(raw_label)
            role = "parent" if label in parent_labels else (
                "reference" if label in all_reference_labels else None
            )
            if role is None:
                continue
            mask = np.asarray(archive["masks"][index], dtype=bool)
            point_indices = lift_mask(projection, sampled_indices, mask)
            if len(point_indices) < int(config["iar"]["minimum_lifted_points"]):
                continue
            stats = frame_completeness(mask)
            confidence = float(archive["scores"][index])
            observations.append({
                "label": label, "role": role,
                "video_id": video_id, "frame_id": frame_id,
                "confidence": confidence,
                "quality": observation_quality(confidence, stats["completeness"]),
                "point_indices": point_indices,
                "centroid": xyz[point_indices].mean(axis=0),
                **stats,
            })
    parents = [row for row in observations if row["role"] == "parent"]
    references = [row for row in observations if row["role"] == "reference"]
    labels = cluster_parent_observations(parents, float(config["iar"]["dbscan_radius_m"]))
    tracks = []
    for track_id in sorted(set(labels.tolist())) if len(labels) else []:
        detections = [row for row in parents if row.get("track_id") == track_id]
        relation_values = []
        supporting_frames = 0
        for detection in detections:
            values = relation_scores(
                contract["external_relation"], detection, references, reference_groups
            )
            if values:
                relation_values.extend(values)
                supporting_frames += 1
        relation_consistency = float(np.mean(relation_values)) if relation_values else 0.0
        attributes = contract["parent"].get("attributes", [])
        attribute_hits = sum(all(attribute in row["label"] for attribute in attributes)
                             for row in detections) if attributes else 0
        attribute_score = min(1.0, attribute_hits / 2.0) if attributes else 0.0
        persistence = min(1.0, len(detections) / 6.0)
        support = min(1.0, supporting_frames / 3.0)
        identity_score = (
            1.5 * attribute_score
            + relation_consistency * (1.0 + support)
            + 0.15 * persistence
        )
        tracks.append({
            "track_id": int(track_id),
            "number_of_detections": len(detections),
            "number_of_relation_frames": supporting_frames,
            "relation_consistency": relation_consistency,
            "identity_score": identity_score,
            "detections": detections,
        })
    selected = max(tracks, key=lambda row: (
        row["identity_score"], row["number_of_detections"]
    )) if tracks else None

    def public_detection(row: dict) -> dict:
        return {
            "video_id": row["video_id"], "frame_id": row["frame_id"],
            "label": row["label"], "confidence": float(row["confidence"]),
            "area": float(row["area"]), "margin": float(row["margin"]),
            "completeness": float(row["completeness"]), "quality": float(row["quality"]),
        }

    return {
        "visit": visit, "desc_id": description_id,
        "instruction": contract.get("query", ""),
        "selected_track": None if selected is None else int(selected["track_id"]),
        "selected_track_detections": (
            [] if selected is None else [public_detection(row) for row in selected["detections"]]
        ),
        "tracks": [
            {key: value for key, value in track.items() if key != "detections"}
            for track in tracks
        ],
        "status": "ok" if selected is not None else "no_parent_track",
    }


def run(config: dict, visits: set[str] | None = None) -> Path:
    contracts_path = Path(config["contracts"])
    payload = json.loads(contracts_path.read_text())
    contracts = [row for row in payload.get("queries", payload)
                 if bool(row.get("requires_instance_resolution"))]
    if visits:
        contracts = [row for row in contracts if str(row["visit"]) in visits]
    overrides = {}
    if config.get("parent_overrides"):
        overrides = json.loads(Path(config["parent_overrides"]).read_text()).get("overrides", {})
    parser = scene_parser(
        path_value(config, "fun3du_root"), path_value(config, "dataset_root"),
        str(config.get("split", "val")),
    )
    visits2videos = visit_to_videos(
        path_value(config, "fun3du_root"), path_value(config, "dataset_root"),
        str(config.get("split", "val")),
    )
    grouped: dict[str, list[dict]] = defaultdict(list)
    for contract in contracts:
        grouped[str(contract["visit"])].append(contract)
    output_dir = Path(config.get("track_dir", Path(config["work_dir"]) / "instance_tracks"))
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for visit, visit_contracts in sorted(grouped.items()):
        xyz, _ = load_scene(parser, visit)
        for index, contract in enumerate(visit_contracts, 1):
            row = build_query_track(
                contract, config=config, parser=parser,
                videos=visits2videos.get(visit, []), xyz=xyz, overrides=overrides,
            )
            rows.append(row)
            case = output_dir / f"{visit}_{row['desc_id']}.json"
            case.write_text(json.dumps(row, indent=2, ensure_ascii=False))
            print(f"[IAR {visit}] {index}/{len(visit_contracts)} "
                  f"{row['desc_id'][:8]} track={row['selected_track']}", flush=True)
    report = Path(config.get("track_report", Path(config["work_dir"]) / "instance_tracks.json"))
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({
        "schema_version": 1,
        "method": "Instance-aware Reweighting: physical instance identification",
        "number_of_queries": len(rows),
        "queries": rows,
    }, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--visits", default="")
    args = parser.parse_args()
    output = run(load_config(args.config), {x for x in args.visits.split(",") if x} or None)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()

