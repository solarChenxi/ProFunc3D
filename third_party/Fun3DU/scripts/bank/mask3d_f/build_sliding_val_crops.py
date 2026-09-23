#!/usr/bin/env python3
"""Precompute SceneFun3D sliding-window validation crops + instance GT.

Writes:
  processed/sliding_crops_validation.yaml
  processed/instance_gt/validation/scene{visit}_{ss:02d}_c{cid:03d}.txt
  processed/sliding_keep/validation/scene..._c....npy  (only if subsampled)

Crop recipe matches train: 4m XY window, stride 2m, keep crops with ≥1 instance
and ≥min_points. Caps at max_points (positives kept, negatives sampled).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml


def _load_yaml(path: Path):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _dump_yaml(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(obj, f, default_flow_style=False, sort_keys=False)


def build_crops(
    processed: Path,
    mode: str = "validation",
    crop_length: float = 4.0,
    crop_stride: float = 2.0,
    min_points: int = 1000,
    max_points: int = 150000,
    require_instance: bool = True,
) -> Path:
    db_path = processed / f"{mode}_database.yaml"
    db = _load_yaml(db_path)
    gt_dir = processed / "instance_gt" / mode
    keep_dir = processed / "sliding_keep" / mode
    gt_dir.mkdir(parents=True, exist_ok=True)
    keep_dir.mkdir(parents=True, exist_ok=True)

    out = []
    n_skip_empty = 0
    n_skip_small = 0
    for item in db:
        pts = np.load(item["filepath"].replace("../../", ""))
        xy = pts[:, :2]
        sem = pts[:, 10].astype(np.int32)
        inst = pts[:, 11].astype(np.int32)
        mn = xy.min(0)
        mx = xy.max(0)
        size = float(crop_length)
        stride = float(crop_stride)
        x0s = np.arange(mn[0], max(mx[0] - size, mn[0]) + 1e-9, stride)
        y0s = np.arange(mn[1], max(mx[1] - size, mn[1]) + 1e-9, stride)
        if x0s.size == 0:
            x0s = np.array([mn[0]], dtype=np.float64)
        if y0s.size == 0:
            y0s = np.array([mn[1]], dtype=np.float64)

        visit = str(item.get("visit", item["scene"]))
        sub_scene = int(item.get("sub_scene", 0))
        cid = 0
        for x0 in x0s:
            for y0 in y0s:
                x1, y1 = float(x0 + size), float(y0 + size)
                chosen = (
                    (xy[:, 0] >= x0)
                    & (xy[:, 0] < x1)
                    & (xy[:, 1] >= y0)
                    & (xy[:, 1] < y1)
                )
                n = int(chosen.sum())
                if n < min_points:
                    n_skip_small += 1
                    continue
                if require_instance and not bool((inst[chosen] > 0).any()):
                    n_skip_empty += 1
                    continue

                idx = np.flatnonzero(chosen)
                keep_path = None
                if idx.size > max_points:
                    rng = np.random.RandomState(
                        (hash((visit, sub_scene, cid)) & 0xFFFFFFFF)
                    )
                    pos = idx[inst[idx] > 0]
                    need = max_points - pos.size
                    if need <= 0:
                        keep = rng.choice(pos, size=max_points, replace=False)
                    else:
                        neg = idx[inst[idx] <= 0]
                        if neg.size > need:
                            neg = rng.choice(neg, size=need, replace=False)
                        keep = np.concatenate([pos, neg]) if neg.size else pos
                    keep = np.sort(keep.astype(np.int64))
                    keep_path = keep_dir / f"scene{visit}_{sub_scene:02d}_c{cid:03d}.npy"
                    np.save(keep_path, keep)
                    idx = keep

                scene_name = f"scene{visit}_{sub_scene:02d}_c{cid:03d}"
                gt_path = gt_dir / f"{scene_name}.txt"
                gt = (
                    sem[idx].astype(np.int32) * 1000
                    + inst[idx].astype(np.int32)
                    + 1
                )
                np.savetxt(gt_path, gt, fmt="%d")

                out.append(
                    {
                        "filepath": item["filepath"],
                        "raw_filepath": item["raw_filepath"],
                        "instance_gt_filepath": str(gt_path.resolve()),
                        "scene": item["scene"],
                        "sub_scene": sub_scene,
                        "visit": visit,
                        "file_len": int(idx.size),
                        "n_instances": int(len(np.unique(inst[idx][inst[idx] > 0]))),
                        "crop_xyxy": [float(x0), float(y0), x1, y1],
                        "scene_name": scene_name,
                        "keep_indices_filepath": (
                            str(keep_path.resolve()) if keep_path is not None else None
                        ),
                        "color_mean": item.get("color_mean"),
                        "color_std": item.get("color_std"),
                    }
                )
                cid += 1

    out_path = processed / f"sliding_crops_{mode}.yaml"
    _dump_yaml(out_path, out)
    print(
        f"wrote {len(out)} crops → {out_path} "
        f"(skip_small={n_skip_small}, skip_empty={n_skip_empty})"
    )
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--processed",
        type=Path,
        default=Path(__file__).resolve().parents[5]
        / "outputs"
        / "val"
        / "mask3d_processed",
    )
    ap.add_argument("--mode", default="validation")
    ap.add_argument("--crop-length", type=float, default=4.0)
    ap.add_argument("--crop-stride", type=float, default=2.0)
    ap.add_argument("--min-points", type=int, default=1000)
    ap.add_argument("--max-points", type=int, default=150000)
    ap.add_argument(
        "--require-instance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="keep only crops that contain a GT instance (off for hidden test)",
    )
    args = ap.parse_args()
    build_crops(
        args.processed,
        mode=args.mode,
        crop_length=args.crop_length,
        crop_stride=args.crop_stride,
        min_points=args.min_points,
        max_points=args.max_points,
        require_instance=args.require_instance,
    )


if __name__ == "__main__":
    main()
