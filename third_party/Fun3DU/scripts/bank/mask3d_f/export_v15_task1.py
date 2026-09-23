#!/usr/bin/env python3
"""Export Mask3D-F ddp8_v15 val30 sliding-window merge preds as Task-1 proposals.

Protocol matches training val:
  198 crops (sliding_crops_validation.yaml) → top-20/crop → class-wise IoU-0.3
  NMS (max 100/scene) via the same sparse-index merge as
  ``trainer._merge_scenefun3d_sliding_preds`` / ``eval_crop_merge_ap.py``.
  ``mask_logit_threshold=2.0`` (v15 hydra).

Writes per-visit ``task1_input.npz`` for ``eval_task1.py`` and optionally
Mask3D instance-eval AP (sanity vs training merge AP).

Usage (Mask3D environment; pick a free GPU):
  CUDA_VISIBLE_DEVICES=0 python -u \\
    third_party/Fun3DU/scripts/bank/mask3d_f/export_v15_task1.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
_s = Path(__file__).resolve().parent
while _s.name != "scripts" and _s.parent != _s:
    _s = _s.parent
if str(_s) not in sys.path:
    sys.path.insert(0, str(_s))
import _pathsetup  # noqa: E402,F401

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[5]
FUN3DU_ROOT = Path(__file__).resolve().parents[3]
MASK3D = REPO_ROOT / "third_party" / "Mask3D"
PROCESSED = REPO_ROOT / "outputs" / "val" / "mask3d_processed"
# Optional proposal-parent metadata. It is absent during standard ProFunc3D
# inference, in which case the exporter simply leaves parent IDs empty.
VAL_TASK1 = REPO_ROOT / "outputs" / "val" / "proposal_parent_metadata"
DEFAULT_CKPT = (
    REPO_ROOT / "checkpoints" / "mask3d_f.ckpt"
)
DEFAULT_OUT = REPO_ROOT / "outputs" / "val" / "proposal_bank"
EVAL_TASK1 = FUN3DU_ROOT / "scripts" / "bank" / "eval_task1.py"

sys.path.insert(0, str(MASK3D))
os.chdir(MASK3D)

from hydra.experimental import compose, initialize  # noqa: E402
import hydra  # noqa: E402
import MinkowskiEngine as ME  # noqa: E402
from trainer.trainer import InstanceSegmentation  # noqa: E402
from benchmark.evaluate_semantic_instance import evaluate  # noqa: E402

KIND_PRIORITY = {
    "furniture": 100,
    "appliance": 90,
    "virtual": 40,
    "landmark": 30,
    "surface": 10,
}


def _load_yaml(path: Path):
    with open(path) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _iou_idx(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    inter = np.intersect1d(a, b, assume_unique=True).size
    if inter == 0:
        return 0.0
    return float(inter) / float(a.size + b.size - inter)


def nms_indices(
    idx_masks: List[np.ndarray],
    scores: List[float],
    classes: List[int],
    n_full: int,
    iou_thr: float = 0.3,
    max_keep: int = 100,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Same class-wise NMS as eval_crop_merge_ap / trainer merge. Returns idx lists."""
    keep_idx, keep_s, keep_c = [], [], []
    by_c: Dict[int, List[int]] = defaultdict(list)
    for i, c in enumerate(classes):
        by_c[int(c)].append(i)
    for c, idxs in by_c.items():
        order = sorted(idxs, key=lambda i: scores[i], reverse=True)
        suppressed = set()
        kept = 0
        for i in order:
            if i in suppressed:
                continue
            keep_idx.append(idx_masks[i])
            keep_s.append(scores[i])
            keep_c.append(c)
            kept += 1
            if kept >= max_keep:
                break
            ai = idx_masks[i]
            for j in order:
                if j == i or j in suppressed:
                    continue
                if _iou_idx(ai, idx_masks[j]) >= iou_thr:
                    suppressed.add(j)
    if not keep_idx:
        return [], np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return (
        keep_idx,
        np.asarray(keep_s, np.float32),
        np.asarray(keep_c, np.int64),
    )


def _predict_crop(model, cfg, batch):
    data, target, file_names = batch
    if cfg.data.add_raw_coordinates:
        raw_coordinates = data.features[:, -3:]
        feats = data.features[:, :-3]
    else:
        raw_coordinates = None
        feats = data.features
    st = ME.SparseTensor(coordinates=data.coordinates, features=feats, device="cuda")
    model.preds = {}
    model.bbox_preds = {}
    model.bbox_gt = {}
    with torch.no_grad():
        out = model.forward(
            st,
            point2segment=[target[i]["point2segment"] for i in range(len(target))],
            raw_coordinates=(
                raw_coordinates.cuda() if raw_coordinates is not None else None
            ),
            is_eval=True,
        )
        model.eval_instance_step(
            out,
            target,
            data.target_full,
            data.inverse_maps,
            file_names,
            data.original_coordinates,
            data.original_colors,
            data.original_normals,
            raw_coordinates,
            data.idx,
        )
    key = file_names[0]
    return key, model.preds[key]


def _dense_from_idx(
    idx_l: Sequence[np.ndarray], n_full: int
) -> np.ndarray:
    if not idx_l:
        return np.zeros((n_full, 0), dtype=bool)
    out = np.zeros((n_full, len(idx_l)), dtype=bool)
    for k, idxs in enumerate(idx_l):
        if idxs.size:
            out[idxs, k] = True
    return out


def save_merged(
    path: Path,
    n_full: int,
    idx_l: List[np.ndarray],
    scores: np.ndarray,
    classes: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = [0]
    chunks = []
    for idxs in idx_l:
        arr = np.asarray(idxs, dtype=np.int64)
        chunks.append(arr)
        offsets.append(offsets[-1] + int(arr.size))
    concat = np.concatenate(chunks) if chunks else np.zeros((0,), dtype=np.int64)
    np.savez_compressed(
        path,
        n_full=np.int64(n_full),
        pred_scores=np.asarray(scores, dtype=np.float32),
        pred_classes=np.asarray(classes, dtype=np.int64),
        idx_concat=concat,
        idx_offsets=np.asarray(offsets, dtype=np.int64),
    )


def load_merged(path: Path) -> Tuple[int, List[np.ndarray], np.ndarray, np.ndarray]:
    z = np.load(path)
    n_full = int(z["n_full"])
    scores = np.asarray(z["pred_scores"], dtype=np.float32)
    classes = np.asarray(z["pred_classes"], dtype=np.int64)
    concat = np.asarray(z["idx_concat"], dtype=np.int64)
    offsets = np.asarray(z["idx_offsets"], dtype=np.int64)
    idx_l = []
    for i in range(len(offsets) - 1):
        idx_l.append(concat[int(offsets[i]) : int(offsets[i + 1])])
    return n_full, idx_l, scores, classes


def load_class_names() -> Tuple[List[str], Dict[int, str]]:
    cmap = json.loads((PROCESSED / "class_map.json").read_text())
    id2name = {int(k): str(v) for k, v in cmap["id2name"].items()}
    names = [id2name[i] for i in sorted(id2name)]
    return names, id2name


def load_gt_instances(visit: str) -> Tuple[List[np.ndarray], List[str], List[str]]:
    """GT instance index lists from processed npy + instance_meta (same space as preds)."""
    meta_p = PROCESSED / "instance_meta" / "validation" / f"{visit}.json"
    npy_p = PROCESSED / "validation" / f"{int(visit):06d}_00.npy"
    pts = np.load(npy_p, mmap_mode="r")
    inst = np.asarray(pts[:, 11], dtype=np.int32)
    sem = np.asarray(pts[:, 10], dtype=np.int32)
    names, id2name = load_class_names()
    _ = names
    idx_l, labels, annot_ids = [], [], []
    if meta_p.is_file():
        meta = json.loads(meta_p.read_text())
        for rec in meta.get("instances", []):
            iid = int(rec["instance_id"])
            m = np.flatnonzero(inst == iid).astype(np.int64)
            if m.size < 5:
                continue
            idx_l.append(m)
            labels.append(str(rec["label"]))
            annot_ids.append(str(rec.get("annot_id", f"{visit}_{iid}")))
        if idx_l:
            return idx_l, labels, annot_ids
    for iid in sorted(int(i) for i in np.unique(inst) if i >= 0):
        m = np.flatnonzero(inst == iid).astype(np.int64)
        if m.size < 5:
            continue
        vals = sem[m]
        vals = vals[vals != 255]
        if vals.size == 0:
            continue
        lab_id = int(np.bincount(vals).argmax())
        if lab_id not in id2name:
            continue
        idx_l.append(m)
        labels.append(id2name[lab_id])
        annot_ids.append(f"{visit}_{iid}")
    return idx_l, labels, annot_ids


def gt_parent_cover(
    visit: str, gt_idx: Sequence[np.ndarray], n_full: int
) -> np.ndarray:
    parents_p = VAL_TASK1 / visit / "parents.json"
    cover = np.full(len(gt_idx), np.nan, dtype=np.float64)
    if not parents_p.is_file():
        return cover
    recs = json.loads(parents_p.read_text()).get("parents", [])
    for rec in recs:
        f = VAL_TASK1 / visit / rec["mask_roi_path"]
        if not f.exists():
            continue
        roi = np.load(f, mmap_mode="r")
        if roi.shape[0] != n_full:
            continue
        roi_idx = np.flatnonzero(np.asarray(roi, dtype=bool)).astype(np.int64)
        for j, gidx in enumerate(gt_idx):
            n = int(gidx.size)
            if n == 0:
                continue
            inter = np.intersect1d(roi_idx, gidx, assume_unique=True).size
            cover[j] = max(cover[j] if np.isfinite(cover[j]) else 0.0, inter / n)
    return cover


def attach_parents(
    visit: str,
    pred_idx: Sequence[np.ndarray],
    n_full: int,
    xyz: Optional[np.ndarray] = None,
) -> Tuple[List[str], List[Dict]]:
    """Nearest parent by pred-mask cover inside ROI, furniture/appliance first."""
    parents_p = VAL_TASK1 / visit / "parents.json"
    parent_ids = [""] * len(pred_idx)
    extras: List[Dict] = [{} for _ in pred_idx]
    if not parents_p.is_file() or not pred_idx:
        return parent_ids, extras
    recs = json.loads(parents_p.read_text()).get("parents", [])
    rois_idx: List[Tuple[str, str, np.ndarray, np.ndarray]] = []
    for rec in recs:
        f = VAL_TASK1 / visit / rec["mask_roi_path"]
        if not f.exists():
            continue
        roi = np.load(f, mmap_mode="r")
        if roi.shape[0] != n_full:
            continue
        kind = rec.get("kind") or ""
        # Landmarks are spatial-reference only; do not hang contact π on them.
        if kind == "landmark":
            continue
        ridx = np.flatnonzero(np.asarray(roi, dtype=bool)).astype(np.int64)
        aabb = rec.get("bbox_aabb") or {}
        mn = np.asarray(aabb.get("min", [0, 0, 0]), dtype=np.float64)
        mx = np.asarray(aabb.get("max", [1, 1, 1]), dtype=np.float64)
        rois_idx.append((rec["parent_id"], kind, ridx, np.stack([mn, mx])))
    for i, pidx in enumerate(pred_idx):
        n = int(pidx.size)
        if n == 0 or not rois_idx:
            continue
        best_s, best_pid, best_kind, best_aabb, best_cover = -1.0, "", "", None, 0.0
        for pid, kind, ridx, aabb in rois_idx:
            inter = np.intersect1d(pidx, ridx, assume_unique=True).size
            cov = inter / n
            pri = KIND_PRIORITY.get(kind, 0) / 100.0
            score = cov + 1e-3 * pri
            if score > best_s:
                best_s, best_pid, best_kind, best_aabb, best_cover = (
                    score, pid, kind, aabb, cov
                )
        parent_ids[i] = best_pid
        rel = [float("nan")] * 3
        if xyz is not None and pidx.size and best_aabb is not None:
            c = np.asarray(xyz[pidx], dtype=np.float64).mean(axis=0)
            span = np.maximum(best_aabb[1] - best_aabb[0], 1e-6)
            rel = ((c - best_aabb[0]) / span).tolist()
        extras[i] = {
            "parent_id": best_pid,
            "parent_kind": best_kind,
            "parent_cover": float(best_cover),
            "rel_aabb_xyz": [float(x) for x in rel],
        }
    return parent_ids, extras


def write_task1_npz(
    visit: str,
    out_dir: Path,
    pred_idx: List[np.ndarray],
    scores: np.ndarray,
    classes_id: np.ndarray,
    n_full: int,
    xyz: Optional[np.ndarray],
) -> Path:
    class_names, id2name = load_class_names()
    name2i = {n: i for i, n in enumerate(class_names)}
    gt_idx, gt_labels, gt_annot = load_gt_instances(visit)
    n_p, n_g = len(pred_idx), len(gt_idx)
    iou = np.zeros((n_p, n_g), dtype=np.float32)
    for i, pidx in enumerate(pred_idx):
        for j, gidx in enumerate(gt_idx):
            iou[i, j] = _iou_idx(pidx, gidx)
    proba = np.zeros((n_p, len(class_names)), dtype=np.float32)
    pred_names = []
    for i, cid in enumerate(classes_id):
        lab = id2name.get(int(cid), "")
        pred_names.append(lab)
        if lab in name2i:
            proba[i, name2i[lab]] = float(scores[i])
    parent_ids, extras = attach_parents(visit, pred_idx, n_full, xyz)
    cover = gt_parent_cover(visit, gt_idx, n_full)
    visit_dir = out_dir / visit
    visit_dir.mkdir(parents=True, exist_ok=True)
    dump = visit_dir / "task1_input.npz"
    np.savez_compressed(
        dump,
        visit=np.asarray(visit),
        classes=np.asarray(class_names),
        proba=proba,
        iou=iou,
        gt_labels=np.asarray(gt_labels),
        gt_annot_ids=np.asarray(gt_annot),
        gt_n=np.asarray([int(g.size) for g in gt_idx], dtype=np.int64),
        gt_parent_cover=cover,
        prop_src=np.asarray(["mask3df_v15"] * n_p),
        prop_parent=np.asarray(parent_ids),
        prop_n=np.asarray([int(p.size) for p in pred_idx], dtype=np.int64),
        prop_class_id=np.asarray(classes_id, dtype=np.int64),
        prop_score=np.asarray(scores, dtype=np.float32),
    )
    proto = []
    for i in range(n_p):
        rec = {
            "src": "mask3df_v15",
            "pred": pred_names[i],
            "conf": float(scores[i]),
            "n_points": int(pred_idx[i].size),
            "class_id": int(classes_id[i]),
        }
        rec.update(extras[i])
        if xyz is not None and pred_idx[i].size:
            rec["centroid"] = np.asarray(xyz[pred_idx[i]], dtype=np.float64).mean(
                axis=0
            ).tolist()
        proto.append(rec)
    with open(visit_dir / "class_bank.json", "w") as f:
        json.dump(
            {
                "visit": visit,
                "n_proposals": n_p,
                "n_gt": n_g,
                "prototypes": proto,
            },
            f,
            indent=2,
        )
    return dump


def compose_v15_cfg():
    conf_rel = os.path.relpath(MASK3D / "conf", Path(__file__).resolve().parent)
    processed = str(PROCESSED)
    with initialize(config_path=conf_rel):
        cfg = compose(
            config_name="config_base_instance_segmentation",
            overrides=[
                "general.experiment_name=export_v15_task1",
                "general.project_name=scenefun3d",
                "general.num_targets=10",
                "general.gpus=1",
                "general.mask_logit_threshold=2.0",
                "data=indoor_scenefun3d",
                "data/datasets=scenefun3d",
                "matcher=hungarian_matcher_mask3df",
                "model.num_queries=80",
                "data.batch_size=1",
                "data.num_workers=0",
                "data.cropping=false",
                "data.validation_dataset.sliding_window=true",
                "data.validation_dataset.crop_stride=2.0",
                f"data.train_dataset.data_dir={processed}",
                f"data.train_dataset.label_db_filepath={processed}/label_database.yaml",
                f"data.train_dataset.color_mean_std={processed}/color_mean_std.yaml",
                f"data.validation_dataset.data_dir={processed}",
                f"data.validation_dataset.label_db_filepath={processed}/label_database.yaml",
                f"data.validation_dataset.color_mean_std={processed}/color_mean_std.yaml",
                "logging=offline",
                "+general.experiment_id=export_v15_task1",
                "+general.version=0",
            ],
        )
    return cfg


def compose_v15_test_cfg():
    conf_rel = os.path.relpath(MASK3D / "conf", Path(__file__).resolve().parent)
    processed = str(PROCESSED)
    with initialize(config_path=conf_rel):
        cfg = compose(
            config_name="config_base_instance_segmentation",
            overrides=[
                "general.experiment_name=export_v15_test",
                "general.project_name=scenefun3d",
                "general.num_targets=10",
                "general.gpus=1",
                "general.mask_logit_threshold=2.0",
                "data=indoor_scenefun3d",
                "data/datasets=scenefun3d",
                "matcher=hungarian_matcher_mask3df",
                "model.num_queries=80",
                "data.batch_size=1",
                "data.num_workers=0",
                "data.cropping=false",
                "data.test_mode=test",
                "data.test_dataset.sliding_window=true",
                "data.test_dataset.crop_stride=2.0",
                f"data.test_dataset.data_dir={processed}",
                f"data.test_dataset.label_db_filepath={processed}/label_database.yaml",
                f"data.test_dataset.color_mean_std={processed}/color_mean_std.yaml",
                "logging=offline",
                "+general.experiment_id=export_v15_test",
                "+general.version=0",
            ],
        )
    return cfg


def infer_visit(
    model,
    cfg,
    val_ds,
    collate,
    name_to_idx,
    crop_list: List[dict],
    top_k: int,
    min_mask_pts: int,
    nms_iou: float,
) -> Tuple[int, List[np.ndarray], np.ndarray, np.ndarray]:
    full_pts = np.load(crop_list[0]["filepath"].replace("../../", ""), mmap_mode="r")
    n_full = int(full_pts.shape[0])
    del full_pts
    idx_l, scores_l, classes_l = [], [], []
    for citem in crop_list:
        sname = citem["scene_name"]
        if sname not in name_to_idx:
            print(f"  skip missing crop {sname}", flush=True)
            continue
        batch = collate([val_ds[name_to_idx[sname]]])
        _, pred = _predict_crop(model, cfg, batch)
        pm = pred["pred_masks"]
        if pm is None or (hasattr(pm, "size") and pm.size == 0):
            continue
        keep_path = citem.get("keep_indices_filepath")
        if keep_path:
            full_idx = np.load(keep_path)
        else:
            x0, y0, x1, y1 = citem["crop_xyxy"]
            pts = np.load(citem["filepath"].replace("../../", ""), mmap_mode="r")
            full_idx = np.flatnonzero(
                (pts[:, 0] >= x0)
                & (pts[:, 0] < x1)
                & (pts[:, 1] >= y0)
                & (pts[:, 1] < y1)
            )
        if len(full_idx) != pm.shape[0]:
            print(
                f"  skip {sname}: mask {pm.shape[0]} vs idx {len(full_idx)}",
                flush=True,
            )
            continue
        order_k = np.argsort(-pred["pred_scores"])[:top_k]
        for k in order_k:
            m_crop = pm[:, int(k)].astype(bool)
            if int(m_crop.sum()) < min_mask_pts:
                continue
            abs_idx = np.unique(full_idx[m_crop].astype(np.int64))
            idx_l.append(abs_idx)
            scores_l.append(float(pred["pred_scores"][int(k)]))
            classes_l.append(int(pred["pred_classes"][int(k)]))
        del pm
    keep_idx, out_s, out_c = nms_indices(
        idx_l, scores_l, classes_l, n_full, iou_thr=nms_iou
    )
    return n_full, keep_idx, out_s, out_c


def run_mask3d_ap(merged_preds: dict, out_dir: Path, ckpt: Path) -> Dict[str, float]:
    gt_path = str(PROCESSED / "instance_gt" / "validation")
    pred_path = str(out_dir / "tmp_output.txt")
    evaluate(merged_preds, gt_path, pred_path, dataset="scenefun3d")
    aps, ap50s, ap25s = [], [], []
    with open(pred_path) as f:
        body = f.read()
    for i, line in enumerate(body.splitlines()):
        if i == 0:
            continue
        parts = line.strip().split(",")
        if len(parts) < 5 or parts[0] in ("mean", "average"):
            continue
        aps.append(float(parts[2]))
        ap50s.append(float(parts[3]))
        ap25s.append(float(parts[4]))
    means = {
        "AP": float(np.nanmean(aps)) if aps else 0.0,
        "AP50": float(np.nanmean(ap50s)) if ap50s else 0.0,
        "AP25": float(np.nanmean(ap25s)) if ap25s else 0.0,
    }
    with open(out_dir / "crop_merge_ap.txt", "w") as f:
        f.write(f"ckpt={ckpt}\n")
        f.write(f"scenes={len(merged_preds)}\n")
        f.write(
            f"AP={means['AP']}\nAP50={means['AP50']}\nAP25={means['AP25']}\n\n"
        )
        f.write(body)
    print("CROP_MERGE_FULL_SCENE_AP", means, flush=True)
    return means


def main() -> None:
    global PROCESSED
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--processed",
        type=Path,
        default=PROCESSED,
        help="Mask3D-F preprocessed dataset produced by prepare_scenefun3d_mask3d.py",
    )
    ap.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--nms-iou", type=float, default=0.3)
    ap.add_argument("--min-mask-pts", type=int, default=10)
    ap.add_argument("--top-k-per-crop", type=int, default=20)
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--visits", default="", help="comma-separated visit ids")
    ap.add_argument("--skip-infer", action="store_true")
    ap.add_argument("--skip-mask3d-ap", action="store_true")
    ap.add_argument("--skip-eval-task1", action="store_true")
    args = ap.parse_args()
    PROCESSED = args.processed.resolve()

    crops = _load_yaml(PROCESSED / "sliding_crops_validation.yaml")
    by_visit: Dict[str, List[dict]] = defaultdict(list)
    for c in crops:
        by_visit[str(c["visit"])].append(c)
    visits = sorted(by_visit.keys())
    if args.visits:
        want = {v.strip() for v in args.visits.split(",") if v.strip()}
        visits = [v for v in visits if v in want]
        missing = want - set(visits)
        if missing:
            raise SystemExit(f"unknown visits: {sorted(missing)}")
    if args.max_scenes is not None:
        visits = visits[: args.max_scenes]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = args.out_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"ckpt={args.ckpt}\nvisits={len(visits)} crops="
        f"{sum(len(by_visit[v]) for v in visits)} out={args.out_dir}",
        flush=True,
    )

    model = cfg = val_ds = collate = name_to_idx = None
    need_infer = [
        v
        for v in visits
        if not args.skip_infer and not (merged_dir / f"{v}.npz").is_file()
    ]
    if need_infer:
        cfg = compose_v15_cfg()
        val_ds = hydra.utils.instantiate(cfg.data.validation_dataset)
        train_ds = hydra.utils.instantiate(cfg.data.train_dataset)
        collate = hydra.utils.instantiate(cfg.data.validation_collation)
        name_to_idx = {
            str(val_ds.data[i].get("scene_name")): i for i in range(len(val_ds.data))
        }
        print(
            f"sliding val crops in dataset: {len(val_ds.data)} "
            f"(expect ~198)",
            flush=True,
        )
        model = InstanceSegmentation(cfg)
        model.validation_dataset = val_ds
        model.train_dataset = train_ds
        state = torch.load(str(args.ckpt), map_location="cpu")
        print(f"loaded epoch={state.get('epoch')} keys={len(state['state_dict'])}", flush=True)
        model.load_state_dict(state["state_dict"], strict=False)
        model = model.cuda().eval()
        del state

    for vi, visit in enumerate(visits):
        mpath = merged_dir / f"{visit}.npz"
        if mpath.is_file():
            n_full, keep_idx, out_s, out_c = load_merged(mpath)
            print(
                f"[{vi+1}/{len(visits)}] {visit}: resume merged "
                f"n={len(keep_idx)} n_full={n_full}",
                flush=True,
            )
        else:
            assert model is not None
            n_full, keep_idx, out_s, out_c = infer_visit(
                model,
                cfg,
                val_ds,
                collate,
                name_to_idx,
                by_visit[visit],
                args.top_k_per_crop,
                args.min_mask_pts,
                args.nms_iou,
            )
            save_merged(mpath, n_full, keep_idx, out_s, out_c)
            print(
                f"[{vi+1}/{len(visits)}] scene{visit}_00: crops={len(by_visit[visit])} "
                f"nms={len(keep_idx)} med_size="
                f"{float(np.median([i.size for i in keep_idx])) if keep_idx else 0:.0f}",
                flush=True,
            )
            torch.cuda.empty_cache()

        npy = np.load(
            PROCESSED / "validation" / f"{int(visit):06d}_00.npy", mmap_mode="r"
        )
        xyz = np.asarray(npy[:, :3])
        write_task1_npz(visit, args.out_dir, keep_idx, out_s, out_c, n_full, xyz)

    if model is not None:
        del model
        torch.cuda.empty_cache()

    means = {}
    if not args.skip_mask3d_ap:
        merged_preds = {}
        for visit in visits:
            n_full, keep_idx, out_s, out_c = load_merged(merged_dir / f"{visit}.npz")
            merged_preds[f"scene{visit}_00"] = {
                "pred_masks": _dense_from_idx(keep_idx, n_full),
                "pred_scores": out_s,
                "pred_classes": out_c,
            }
        means = run_mask3d_ap(merged_preds, args.out_dir, args.ckpt)
        del merged_preds

    eval_json = args.out_dir / "task1_eval.json"
    if not args.skip_eval_task1:
        import subprocess

        cmd = [
            sys.executable,
            str(EVAL_TASK1),
            "--runs",
            str(args.out_dir / "*/task1_input.npz"),
            "--out",
            str(eval_json),
        ]
        print("[eval_task1]", " ".join(cmd), flush=True)
        subprocess.check_call(cmd, cwd=str(FUN3DU_ROOT))
        report = json.loads(eval_json.read_text())
        b25 = report["by_thr"]["0.25"]
        b50 = report["by_thr"]["0.5"]
        r25 = b25["recall_micro"]["recall_agnostic"]
        r50 = b50["recall_micro"]["recall_agnostic"]
        summary = {
            "ckpt": str(args.ckpt),
            "protocol": "val30 sliding-window 198 crops → class-wise NMS IoU0.3 max100; mask_logit_threshold=2.0",
            "n_visits": report["n_visits"],
            "n_gt": report["n_gt"],
            "n_proposals": report["n_proposals"],
            "mask3d_merge_ap": means,
            "task1": {
                "mAP@0.25": b25["mAP"],
                "mAP@0.5": b50["mAP"],
                "R_agn@0.25": r25,
                "R_agn@0.5": r50,
                "R_top1@0.25": b25["recall_micro"]["recall_top1"],
                "R_top1@0.5": b50["recall_micro"]["recall_top1"],
            },
            "old_bank": {"mAP@0.25": 0.0455, "R_agn@0.25": 0.3458},
            "gate_B": {
                "R_agn@0.25_vs_0.35": r25,
                "target_R_agn": 0.55,
                "pass": bool(r25 >= 0.55 and b25["mAP"] > 0.0455),
            },
        }
        with open(args.out_dir / "idea_b_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print("\n===== Idea B gate =====", flush=True)
        print(
            f"  Mask3D merge AP/AP50/AP25 = "
            f"{means.get('AP', float('nan')):.3f}/"
            f"{means.get('AP50', float('nan')):.3f}/"
            f"{means.get('AP25', float('nan')):.3f}",
            flush=True,
        )
        print(
            f"  Task-1 mAP@0.25={b25['mAP']:.4f} (old bank 0.0455)  "
            f"mAP@0.5={b50['mAP']:.4f} (old 0.0270)",
            flush=True,
        )
        print(
            f"  R_agn@0.25={r25:.4f} (old 0.35, gate ≥0.55)  "
            f"R_agn@0.5={r50:.4f}",
            flush=True,
        )
        print(f"  PASS={summary['gate_B']['pass']}", flush=True)
        print(f"[OK] {args.out_dir / 'idea_b_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
