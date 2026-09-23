"""Retrieve parent/reference observations required by relational IAR queries."""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .backend import activate_fun3du, load_grounding_observations, scene_parser
from .config import load_config, path_value


def pad_strings(rows: list[list[str]]) -> np.ndarray:
    width = max((len(row) for row in rows), default=1)
    return np.asarray([row + ["nan"] * (width - len(row)) for row in rows])


def frame_ranges(grounding: dict | None, context_s: float) -> dict[str, list[tuple[float, float]]]:
    ranges: dict[str, list[tuple[float, float]]] = defaultdict(list)
    if grounding is None:
        return ranges
    videos = grounding["video_ids"][:50]
    frames = grounding["frame_ids"][:50]
    for video_id in set(map(str, videos)):
        timestamps = [float(frame) for video, frame in zip(videos, frames)
                      if str(video) == video_id]
        if timestamps:
            ranges[video_id].append((min(timestamps) - context_s,
                                     max(timestamps) + context_s))
    return ranges


def select_frames(parser, visit: str, ranges: dict, gap_s: float) -> list[tuple[str, str, str]]:
    selected = []
    for video_id, intervals in sorted(ranges.items()):
        candidates = [
            (float(frame_id), str(frame_id), path)
            for frame_id, path in parser.get_rgb_frames(visit, video_id).items()
            if any(start <= float(frame_id) <= end for start, end in intervals)
        ]
        previous = -float("inf")
        for timestamp, frame_id, path in sorted(candidates):
            if timestamp - previous >= gap_s:
                selected.append((video_id, frame_id, path))
                previous = timestamp
    return selected


def merge_detections(path: Path, records: list[dict], mask_score) -> list[str]:
    if not records:
        return []
    old = dict(np.load(path, allow_pickle=True)) if path.is_file() else {}
    new_masks = [np.asarray(mask, np.uint8) for record in records for mask in record["masks"]]
    new_labels = [str(label) for record in records for label in record["labels"]]
    new_scores = [float(score) for record in records for score in record["scores"]]
    if not new_masks:
        return []
    if old:
        masks = list(old["masks"]) + new_masks
        labels = [str(value) for value in old["labels"].tolist()] + new_labels
        scores = [float(value) for value in old["scores"].tolist()] + new_scores
        modification = [float(value) for value in old["mod_scores"].tolist()]
        angle = [float(value) for value in old["angle_scores"].tolist()]
        descriptions = [
            [str(value) for value in np.asarray(row).reshape(-1).tolist()]
            for row in old["desc_ids"]
        ]
    else:
        masks, labels, scores = new_masks, new_labels, new_scores
        modification, angle, descriptions = [], [], []
    for mask in masks[len(modification):]:
        angle_score, modification_score = mask_score(torch.tensor(mask))
        angle.append(float(angle_score))
        modification.append(float(modification_score))
    descriptions += [["nan"] for _ in range(len(labels) - len(descriptions))]
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            masks=np.asarray(masks, np.uint8),
            labels=np.asarray(labels),
            scores=np.asarray(scores),
            mod_scores=np.asarray(modification),
            angle_scores=np.asarray(angle),
            desc_ids=pad_strings(descriptions),
        )
    os.replace(temporary, path)
    return new_labels


def run(config: dict, visits: set[str] | None = None) -> None:
    fun3du_root = path_value(config, "fun3du_root")
    activate_fun3du(fun3du_root)
    from utils.hf_models import init_detection, process_detection
    from utils.metrics import get_mask_score

    dataset_root = path_value(config, "dataset_root")
    split = str(config.get("split", "val"))
    work_dir = path_value(config, "work_dir")
    mask_type = str(config.get("object_masks", "owl2_rsam_v2"))
    contracts_payload = json.loads(Path(config["contracts"]).read_text())
    contracts = contracts_payload.get("queries", contracts_payload)
    contracts = [row for row in contracts if row.get("requires_instance_resolution")]
    if visits:
        contracts = [row for row in contracts if str(row["visit"]) in visits]
    tasks: dict[str, dict] = defaultdict(lambda: {"labels": set(), "descriptions": set()})
    for contract in contracts:
        visit = str(contract["visit"])
        labels = list(contract["parent"].get("retrieval_keys", []))
        for reference in contract.get("anchors", []):
            labels.extend(reference.get("retrieval_keys", []))
        tasks[visit]["labels"].update(str(label) for label in labels if label)
        tasks[visit]["descriptions"].add(str(contract["desc_id"]))
    parser = scene_parser(fun3du_root, dataset_root, split)
    detector, detector_processor, segmenter, segmenter_processor = init_detection(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    grounding_dir = Path(config["grounding_observations"])
    context_s = float(config["iar"]["context_window_s"])
    gap_s = float(config["iar"]["timeline_sampling_s"])
    chunk_size = int(config.get("retrieval_label_chunk", 6))
    for task_index, (visit, task) in enumerate(sorted(tasks.items()), 1):
        ranges: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for description_id in task["descriptions"]:
            current = frame_ranges(
                load_grounding_observations(
                    grounding_dir / f"{visit}_{description_id}.npz"
                ), context_s
            )
            for video_id, intervals in current.items():
                ranges[video_id].extend(intervals)
        frames = select_frames(parser, visit, ranges, gap_s)
        index_path = dataset_root / split / visit / f"{visit}_{mask_type}_masks.json"
        index = json.loads(index_path.read_text()) if index_path.is_file() else {
            "desc_ids": {}, "objects": {}
        }
        print(f"[relation retrieval {task_index}/{len(tasks)}] visit={visit} "
              f"labels={len(task['labels'])} frames={len(frames)}", flush=True)
        for frame_index, (video_id, frame_id, image_path) in enumerate(frames, 1):
            output = dataset_root / split / visit / video_id / mask_type / f"{video_id}_{frame_id}.npz"
            present = set()
            if output.is_file():
                with np.load(output, allow_pickle=True) as archive:
                    present = {str(value) for value in archive["labels"].tolist()}
            pending = sorted(set(task["labels"]) - present)
            records = []
            if pending:
                image = Image.open(image_path).convert("RGB")
                for start in range(0, len(pending), chunk_size):
                    result = process_detection(
                        detector, detector_processor, segmenter, segmenter_processor,
                        [image], pending[start:start + chunk_size],
                    )[0]
                    if isinstance(result["labels"], np.ndarray):
                        records.append(result)
                    torch.cuda.empty_cache()
            added = merge_detections(output, records, get_mask_score)
            key = f"{video_id} {frame_id}"
            for label in present | set(added):
                if label in task["labels"]:
                    index["objects"].setdefault(label, [])
                    if key not in index["objects"][label]:
                        index["objects"][label].append(key)
            if frame_index % 20 == 0 or frame_index == len(frames):
                print(f"  {frame_index}/{len(frames)}", flush=True)
        temporary = index_path.with_suffix(index_path.suffix + f".tmp.{os.getpid()}")
        temporary.write_text(json.dumps(index))
        os.replace(temporary, index_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--visits", default="")
    args = parser.parse_args()
    run(load_config(args.config), {value for value in args.visits.split(",") if value} or None)


if __name__ == "__main__":
    main()

