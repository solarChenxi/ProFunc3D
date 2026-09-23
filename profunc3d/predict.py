"""Select and export the final ProFunc3D functional mask."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .backend import load_proposals, load_scene, scene_parser, write_ascii_ply
from .config import load_config, path_value
from .iar import predict as aggregate, recover_paper_track


def load_rows(path: Path) -> list[dict]:
    payload = json.loads(path.read_text())
    return payload.get("queries", payload if isinstance(payload, list) else [payload])


def load_cases(directory: Path | None) -> dict[tuple[str, str], dict]:
    cases = {}
    if directory is None or not directory.is_dir():
        return cases
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text())
        for row in payload.get("queries", [payload]):
            if "visit" in row and "desc_id" in row:
                cases[(str(row["visit"]), str(row["desc_id"]))] = row
    return cases


def run(config: dict) -> Path:
    work_dir = path_value(config, "work_dir")
    contracts = {
        (str(row["visit"]), str(row["desc_id"])): row
        for row in load_rows(Path(config["contracts"]))
    }
    associations = {
        (str(row["visit"]), str(row["desc_id"])): row
        for row in load_rows(Path(config.get("pga_output", work_dir / "pga_associations.json")))
    }
    track_dir = Path(config.get("track_dir", work_dir / "instance_tracks"))
    tracks = load_cases(track_dir)
    frozen_dir = Path(config["frozen_track_dir"]) if config.get("frozen_track_dir") else None
    frozen_tracks = load_cases(frozen_dir)
    output_dir = Path(config.get("prediction_dir", work_dir / "predictions"))
    output_dir.mkdir(parents=True, exist_ok=True)
    parser = scene_parser(
        path_value(config, "fun3du_root"), path_value(config, "dataset_root"),
        str(config.get("split", "val")),
    )
    bank_root = path_value(config, "proposal_bank")
    scene_cache = {}
    summary = []
    for key, association in sorted(associations.items()):
        if key not in contracts:
            raise RuntimeError(f"missing instruction contract for {key}")
        visit, description_id = key
        if visit not in scene_cache:
            xyz, rgb = load_scene(parser, visit)
            proposals = load_proposals(bank_root, visit, len(xyz))
            scene_cache[visit] = xyz, rgb, proposals
        xyz, rgb, proposals = scene_cache[visit]
        contract = contracts[key]
        track = tracks.get(key, {})
        if frozen_dir is not None:
            detections = recover_paper_track(track, frozen_tracks.get(key, {}))
            track_source = "paper_checkpoint_compatibility"
        else:
            detections = list(track.get("selected_track_detections", []))
            # Accept the serialized name used by the frozen development cache.
            if not detections:
                detections = list(track.get("chosen_detections", []))
            track_source = "fresh_instance_identification"
        parameters = config["pga"] | config["iar"]
        decision = aggregate(
            observations=list(association.get("observations", [])),
            number_of_proposals=len(proposals),
            relational=bool(contract.get("requires_instance_resolution")),
            selected_track_detections=detections,
            generic_support_ratio=float(config["pga"]["generic_support_ratio"]),
            relational_support_ratio=float(config["pga"]["relational_support_ratio"]),
            temporal_window_s=float(config["iar"]["temporal_alignment_s"]),
            beta=float(config["iar"]["beta"]),
            gamma=float(config["iar"]["gamma"]),
        )
        selected_proposal = int(decision["selected_proposal"])
        selected_indices = (
            proposals[selected_proposal]
            if 0 <= selected_proposal < len(proposals)
            else np.zeros(0, dtype=np.int64)
        )
        point_mask = np.zeros(len(xyz), dtype=bool)
        point_mask[selected_indices] = True
        query_dir = output_dir / f"{visit}_{description_id}"
        query_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            query_dir / "prediction.npz",
            point_mask=point_mask,
            selected_indices=selected_indices.astype(np.int64),
            selected_proposal=np.int64(selected_proposal),
        )
        metadata = {
            "visit": visit,
            "desc_id": description_id,
            "instruction": contract.get("query", association.get("instruction", "")),
            "relational": bool(contract.get("requires_instance_resolution")),
            "track_source": track_source,
            "number_of_track_detections": len(detections),
            "number_of_scene_points": len(xyz),
            "number_of_selected_points": int(point_mask.sum()),
            "parameters": parameters,
            **decision,
        }
        (query_dir / "prediction.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False)
        )
        if bool(config.get("write_ply", True)):
            selected_ids = np.flatnonzero(point_mask)
            background_ids = np.flatnonzero(~point_mask)
            cap = int(config.get("visualization_points", 300000))
            budget = max(0, cap - len(selected_ids))
            if len(background_ids) > budget:
                step = len(background_ids) / max(1, budget)
                background_ids = background_ids[(np.arange(budget) * step).astype(np.int64)]
            visual_ids = np.concatenate([background_ids, selected_ids])
            colors = np.asarray(rgb, dtype=np.float64) * 0.38 + 0.18
            colors[point_mask] = np.asarray([1.0, 0.32, 0.02])
            write_ascii_ply(
                query_dir / "scene_with_prediction.ply", xyz[visual_ids], colors[visual_ids]
            )
            if selected_ids.size:
                write_ascii_ply(
                    query_dir / "predicted_region.ply",
                    xyz[selected_ids],
                    np.tile(np.asarray([[1.0, 0.32, 0.02]]), (len(selected_ids), 1)),
                )
        summary.append({
            "visit": visit, "desc_id": description_id,
            "route": decision["route"],
            "selected_proposal": selected_proposal,
            "number_of_selected_points": int(point_mask.sum()),
            "output_dir": str(query_dir),
        })
        print(f"[ProFunc3D {visit}] {description_id[:8]} "
              f"proposal={selected_proposal} route={decision['route']}", flush=True)
    manifest = output_dir / "predictions.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "method": "ProFunc3D",
        "number_of_queries": len(summary),
        "queries": summary,
    }, indent=2, ensure_ascii=False))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    output = run(load_config(args.config))
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()

