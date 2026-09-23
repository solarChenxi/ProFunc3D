#!/usr/bin/env python3
"""Convert SceneFun3D laser + affordance GT into Mask3D processed format (P0).

Each scene becomes a float32 `.npy` with columns matching Mask3D/ScanNet:
    [x y z | r g b | nx ny nz | segment | semantic | instance]
where colors are 0..255, background semantic=255 / instance=-1, and the
9 affordance classes are ids 1..9 (0 unused).

Also writes:
  - instance_gt/{mode}/sceneXXXX_00.txt   (semantic*1000 + instance + 1)
  - {mode}_database.yaml
  - label_database.yaml
  - color_mean_std.yaml
  - splits.json (usable vs skipped visits)

Usage:
  python \\
    scripts/bank/mask3d_f/prepare_scenefun3d_mask3d.py \\
      --splits train,val --max-visits 0 \\
      --out-dir exps/prototype_bank/mask3d_f/processed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import yaml
from scipy.spatial import cKDTree

_ROOT = Path(__file__).resolve().parents[3]
_s = Path(__file__).resolve().parent
while _s.name != "scripts" and _s.parent != _s:
    _s = _s.parent
if str(_s) not in sys.path:
    sys.path.insert(0, str(_s))
import _pathsetup  # noqa: E402,F401
_PB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_PB))

from io_scene import DEFAULT_DATA_ROOT, load_cropped_xyz, load_gt_annots  # noqa: E402
from utils.sun3d.data_parser import DataParser  # noqa: E402

# Closed vocabulary for SceneFun3D Task-1 (exclude is ignored).
CLASS_NAMES = [
    "rotate",
    "key_press",
    "tip_push",
    "hook_turn",
    "pinch_pull",
    "hook_pull",
    "plug_in",
    "unplug",
    "foot_push",
]
# Mask3D convention: id 0 unused/empty; 1..K are valid classes; 255 = ignore.
NAME2ID = {n: i + 1 for i, n in enumerate(CLASS_NAMES)}
IGNORE_SEMANTIC = 255
COLORS = [
    [230, 25, 75],
    [60, 180, 75],
    [255, 225, 25],
    [0, 130, 200],
    [245, 130, 48],
    [145, 30, 180],
    [70, 240, 240],
    [240, 50, 230],
    [210, 245, 60],
]


def _list_split_visits(data_root: Path, split: str) -> List[str]:
    f = data_root / "benchmark_file_lists" / f"{split}_scenes.txt"
    if not f.exists():
        return sorted(
            d.name for d in (data_root / split).iterdir() if d.is_dir() and d.name.isdigit()
        )
    return [ln.strip() for ln in f.read_text().splitlines() if ln.strip()]


def _visit_ok(
    data_root: Path, split: str, visit: str, *, require_annotations: bool = True
) -> bool:
    d = data_root / split / visit
    ok = (d / f"{visit}_laser_scan.ply").is_file() and (
        d / f"{visit}_crop_mask.npy"
    ).is_file()
    if require_annotations:
        ok = ok and (d / f"{visit}_annotations.json").is_file()
    return ok


def _estimate_normals(
    xyz: np.ndarray,
    voxel: float = 0.04,
    max_nn: int = 30,
) -> np.ndarray:
    """Fast normals via voxel-downsample estimate + NN map-back.

    Avoids Open3D ``orient_normals_consistent_tangent_plane`` on multi-million
    laser clouds (minutes→stuck). Voxel 4 cm ≈ 0.5 s for ~2.5M pts.
    """
    if len(xyz) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    if voxel > 0 and len(xyz) > 50_000:
        down = pcd.voxel_down_sample(voxel_size=voxel)
        radius = max(voxel * 2.5, 0.05)
        down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius, max_nn=max_nn
            )
        )
        down.orient_normals_towards_camera_location(pcd.get_center())
        dp = np.asarray(down.points)
        dn = np.asarray(down.normals)
        tree = cKDTree(dp)
        _, idx = tree.query(xyz, k=1, workers=-1)
        return dn[idx].astype(np.float32)
    # small clouds: direct knn, no expensive consistent-tangent orient
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=max_nn)
    )
    pcd.orient_normals_towards_camera_location(pcd.get_center())
    return np.asarray(pcd.normals, dtype=np.float32)


def _build_labels(
    n: int, gts: List[Dict]
) -> Tuple[np.ndarray, np.ndarray, List[Dict], int]:
    """Return semantic (N,), instance (N,), per-instance meta, n_overlap_pts."""
    semantic = np.full(n, IGNORE_SEMANTIC, dtype=np.int32)
    instance = np.full(n, -1, dtype=np.int32)
    metas: List[Dict] = []
    claimed = np.zeros(n, dtype=bool)
    overlap = 0
    for g in gts:
        lab = g["label"]
        if lab not in NAME2ID or g["gt_n"] < 5:
            continue
        mask = g["gt"].astype(bool)
        overlap += int((claimed & mask).sum())
        # later annotations overwrite on conflict (rare)
        semantic[mask] = NAME2ID[lab]
        inst_id = len(metas)
        instance[mask] = inst_id
        claimed[mask] = True
        metas.append(
            {
                "instance_id": inst_id,
                "label": lab,
                "class_id": NAME2ID[lab],
                "annot_id": g["annot_id"],
                "n_points": int(mask.sum()),
            }
        )
    return semantic, instance, metas, overlap


def _scene_paths(out_dir: Path, mode: str, visit: str) -> Dict[str, Path]:
    scene_id = int(visit)
    return {
        "npy": out_dir / mode / f"{scene_id:06d}_00.npy",
        "gt": out_dir / "instance_gt" / mode / f"scene{scene_id:06d}_00.txt",
        "meta": out_dir / "instance_meta" / mode / f"{visit}.json",
    }


def _filebase_from_existing(
    parser: DataParser,
    split: str,
    visit: str,
    paths: Dict[str, Path],
) -> Optional[Dict]:
    """Rebuild database entry from a previously written visit (resume)."""
    if not (paths["npy"].is_file() and paths["gt"].is_file() and paths["meta"].is_file()):
        return None
    try:
        meta = json.loads(paths["meta"].read_text())
        # color stats from a light mmap slice of rgb columns
        pts = np.load(paths["npy"], mmap_mode="r")
        n = int(pts.shape[0])
        # sample for color mean/std to avoid reading full cloud
        step = max(n // 200_000, 1)
        col = np.asarray(pts[::step, 3:6], dtype=np.float64) / 255.0
        # DataParser.data_root_path already includes split — do NOT join split again.
        raw = Path(parser.data_root_path) / visit / f"{visit}_laser_scan.ply"
        return {
            "filepath": str(paths["npy"].resolve()),
            "raw_filepath": str(raw.resolve()),
            "instance_gt_filepath": str(paths["gt"].resolve()),
            "scene": int(visit),
            "sub_scene": 0,
            "visit": visit,
            "file_len": n,
            "n_instances": int(meta.get("n_instances", 0)),
            "color_mean": [
                float(col[:, 0].mean()),
                float(col[:, 1].mean()),
                float(col[:, 2].mean()),
            ],
            "color_std": [
                float((col[:, 0] ** 2).mean()),
                float((col[:, 1] ** 2).mean()),
                float((col[:, 2] ** 2).mean()),
            ],
        }
    except Exception as e:
        print(f"  [{visit}] resume-load fail: {e}", flush=True)
        return None


def process_visit(
    parser: DataParser,
    split: str,
    visit: str,
    out_dir: Path,
    mode: str,
    normal_voxel: float,
    resume: bool,
    force: bool,
    allow_no_gt: bool = False,
) -> Optional[Dict]:
    paths = _scene_paths(out_dir, mode, visit)
    if resume and not force:
        fb = _filebase_from_existing(parser, split, visit, paths)
        if fb is not None:
            print(
                f"  [{visit}] resume skip pts={fb['file_len']} inst={fb['n_instances']}",
                flush=True,
            )
            return fb

    t0 = time.time()
    try:
        xyz, rgb, _ = load_cropped_xyz(parser, visit)
        gts = [] if allow_no_gt else load_gt_annots(parser, visit, skip_exclude=True)
    except Exception as e:
        print(f"  [{visit}] load fail: {e}", flush=True)
        return None
    if len(xyz) < 1000:
        print(f"  [{visit}] too few points ({len(xyz)})", flush=True)
        return None

    semantic, instance, metas, overlap = _build_labels(len(xyz), gts)
    if not metas and not allow_no_gt:
        print(f"  [{visit}] no usable instances", flush=True)
        return None

    # colors 0..255 for Mask3D
    col = rgb.astype(np.float64)
    if col.max() <= 1.5:
        col = col * 255.0
    col = np.clip(col, 0, 255).astype(np.float32)

    t_n0 = time.time()
    normals = _estimate_normals(xyz, voxel=normal_voxel)
    t_n1 = time.time()
    segments = np.maximum(instance, 0).astype(np.float32)  # bg → 0

    points = np.hstack(
        [
            xyz.astype(np.float32),
            col,
            normals,
            segments[:, None],
            semantic.astype(np.float32)[:, None],
            instance.astype(np.float32)[:, None],
        ]
    )
    assert points.shape[1] == 12

    paths["npy"].parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["npy"], points)

    gt_data = semantic.astype(np.int32) * 1000 + instance.astype(np.int32) + 1
    paths["gt"].parent.mkdir(parents=True, exist_ok=True)
    # buffered text write (faster than default savetxt buffering on huge clouds)
    with open(paths["gt"], "w", buffering=8 * 1024 * 1024) as f:
        np.savetxt(f, gt_data, fmt="%d")

    paths["meta"].parent.mkdir(parents=True, exist_ok=True)
    with open(paths["meta"], "w") as f:
        json.dump(
            {
                "visit": visit,
                "split": split,
                "n_points": int(len(xyz)),
                "n_instances": len(metas),
                "n_overlap_points": int(overlap),
                "normal_voxel": float(normal_voxel),
                "instances": metas,
            },
            f,
            indent=2,
        )

    color01 = col / 255.0
    # DataParser.data_root_path already includes split — do NOT join split again.
    raw = Path(parser.data_root_path) / visit / f"{visit}_laser_scan.ply"
    filebase = {
        "filepath": str(paths["npy"].resolve()),
        "raw_filepath": str(raw.resolve()),
        "instance_gt_filepath": str(paths["gt"].resolve()),
        "scene": int(visit),
        "sub_scene": 0,
        "visit": visit,
        "file_len": int(len(xyz)),
        "n_instances": len(metas),
        "color_mean": [
            float(color01[:, 0].mean()),
            float(color01[:, 1].mean()),
            float(color01[:, 2].mean()),
        ],
        "color_std": [
            float((color01[:, 0] ** 2).mean()),
            float((color01[:, 1] ** 2).mean()),
            float((color01[:, 2] ** 2).mean()),
        ],
    }
    print(
        f"  [{visit}] pts={len(xyz)} inst={len(metas)} overlap_pts={overlap} "
        f"normals={t_n1 - t_n0:.1f}s total={time.time() - t0:.1f}s -> {paths['npy'].name}",
        flush=True,
    )
    return filebase


def write_label_database(out_dir: Path) -> None:
    db = {
        0: {"name": "empty", "color": [0, 0, 0], "validation": False},
    }
    for name, cid in NAME2ID.items():
        db[cid] = {
            "name": name,
            "color": COLORS[cid - 1],
            "validation": True,
        }
    with open(out_dir / "label_database.yaml", "w") as f:
        yaml.safe_dump(db, f, sort_keys=True)


def write_color_mean_std(out_dir: Path, train_db: List[Dict]) -> None:
    if not train_db:
        return
    means = np.array([s["color_mean"] for s in train_db], dtype=np.float64)
    secs = np.array([s["color_std"] for s in train_db], dtype=np.float64)
    m = means.mean(axis=0)
    # color_std in filebase stores E[x^2]; convert to std like Mask3D
    std = np.sqrt(np.clip(secs.mean(axis=0) - m**2, 0, None))
    with open(out_dir / "color_mean_std.yaml", "w") as f:
        yaml.safe_dump(
            {"mean": [float(x) for x in m], "std": [float(x) for x in std]},
            f,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--splits", default="train,val")
    ap.add_argument(
        "--allow-no-gt",
        action="store_true",
        help="inference-only (hidden test): dummy labels, no annotations.json",
    )
    ap.add_argument("--max-visits", type=int, default=0, help="0=all; per split")
    ap.add_argument(
        "--out-dir",
        default="exps/prototype_bank/mask3d_f/processed",
    )
    ap.add_argument(
        "--visits",
        default="",
        help="optional comma visit ids (overrides split listing; still need --splits for mode)",
    )
    ap.add_argument(
        "--normal-voxel",
        type=float,
        default=0.04,
        help="voxel size for normal estimate (0=full-res knn)",
    )
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip visits that already have npy+gt+meta; rebuild database (default: on)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="recompute even if resume artifacts exist",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_label_database(out_dir)

    split_map = {
        "train": "train",
        "val": "validation",
        "validation": "validation",
        "test": "test",
    }
    want_splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    forced = [v.strip() for v in args.visits.split(",") if v.strip()]

    summary = {
        "data_root": str(data_root),
        "classes": CLASS_NAMES,
        "normal_voxel": args.normal_voxel,
        "splits": {},
    }
    train_db_for_stats: List[Dict] = []

    for split in want_splits:
        mode = split_map.get(split, split)
        data_split = "val" if split in ("val", "validation") else split
        parser = DataParser(str(data_root), data_split)
        if forced:
            visits = forced
        else:
            visits = _list_split_visits(data_root, data_split)
        if args.max_visits > 0:
            visits = visits[: args.max_visits]
            existing_db = out_dir / f"{mode}_database.yaml"
            if existing_db.is_file():
                try:
                    prev = yaml.safe_load(existing_db.read_text()) or []
                    if len(prev) > len(visits):
                        print(
                            f"[warn] --max-visits={args.max_visits} will rewrite "
                            f"{existing_db.name} ({len(prev)} → ≤{len(visits)} entries). "
                            f"Re-run without --max-visits to restore the full database.",
                            flush=True,
                        )
                except Exception:
                    pass

        usable, skipped = [], []
        database: List[Dict] = []
        print(
            f"\n=== {split} → mode={mode} candidates={len(visits)} "
            f"resume={args.resume} normal_voxel={args.normal_voxel}",
            flush=True,
        )
        for i, visit in enumerate(visits, 1):
            print(f"[{i}/{len(visits)}] {visit}", flush=True)
            if not _visit_ok(
                data_root,
                data_split,
                visit,
                require_annotations=not args.allow_no_gt,
            ):
                skipped.append({"visit": visit, "reason": "missing_files"})
                print(f"  [{visit}] missing_files", flush=True)
                continue
            fb = process_visit(
                parser,
                data_split,
                visit,
                out_dir,
                mode,
                normal_voxel=args.normal_voxel,
                resume=args.resume,
                force=args.force,
                allow_no_gt=args.allow_no_gt,
            )
            if fb is None:
                skipped.append({"visit": visit, "reason": "process_fail"})
                continue
            database.append(fb)
            usable.append(visit)

        db_path = out_dir / f"{mode}_database.yaml"
        with open(db_path, "w") as f:
            yaml.safe_dump(database, f, sort_keys=False)
        if mode == "train":
            train_db_for_stats = database
            with open(out_dir / "train_database.yaml", "w") as f:
                yaml.safe_dump(database, f, sort_keys=False)
        if mode == "validation":
            with open(out_dir / "validation_database.yaml", "w") as f:
                yaml.safe_dump(database, f, sort_keys=False)
        if mode == "test":
            with open(out_dir / "test_database.yaml", "w") as f:
                yaml.safe_dump(database, f, sort_keys=False)

        n_inst = sum(x["n_instances"] for x in database)
        n_pts = sum(x["file_len"] for x in database)
        summary["splits"][mode] = {
            "n_usable": len(usable),
            "n_skipped": len(skipped),
            "n_instances": n_inst,
            "n_points": n_pts,
            "usable": usable,
            "skipped": skipped,
            "database": str(db_path),
        }
        print(
            f"[done] {mode}: usable={len(usable)} skipped={len(skipped)} "
            f"instances={n_inst} pts={n_pts}",
            flush=True,
        )

    if train_db_for_stats:
        write_color_mean_std(out_dir, train_db_for_stats)
    splits_name = "splits_test.json" if want_splits == ["test"] else "splits.json"
    with open(out_dir / splits_name, "w") as f:
        json.dump(summary, f, indent=2)
    cmap_path = out_dir / "class_map.json"
    if not cmap_path.is_file() or "train" in want_splits or "val" in want_splits:
        with open(cmap_path, "w") as f:
            json.dump(
                {
                    "name2id": NAME2ID,
                    "id2name": {v: k for k, v in NAME2ID.items()},
                    "ignore": IGNORE_SEMANTIC,
                },
                f,
                indent=2,
            )
    print(f"\n[OK] {out_dir}", flush=True)


if __name__ == "__main__":
    main()
