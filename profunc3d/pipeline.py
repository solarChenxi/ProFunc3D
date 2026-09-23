"""End-to-end ProFunc3D command line pipeline."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

from .associate import run as run_pga
from .config import load_config, path_value
from .predict import run as run_prediction
from .relation_retrieval import run as run_relation_retrieval
from .tracks import run as run_instance_identification

STAGES = (
    "instruction_decomposition",
    "parent_retrieval",
    "sparse_grounding",
    "contracts",
    "relation_retrieval",
    "proposal_bank",
    "instance_identification",
    "proposal_grounded_association",
    "prediction",
)


def run_command(command: list[str], cwd: Path, dry_run: bool) -> None:
    print(shlex.join(command), flush=True)
    if not dry_run:
        subprocess.check_call(command, cwd=str(cwd))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stages", default=",".join(STAGES))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    selected_stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    unknown = [stage for stage in selected_stages if stage not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stages: {unknown}; choices: {', '.join(STAGES)}")

    fun3du = path_value(config, "fun3du_root")
    python = str(Path(config.get("python", sys.executable)).expanduser())
    mask3d_python = str(Path(config.get("mask3d_python", python)).expanduser())
    dataset_root = path_value(config, "dataset_root")
    split = str(config.get("split", "val"))
    work_dir = path_value(config, "work_dir")
    if not args.dry_run:
        work_dir.mkdir(parents=True, exist_ok=True)
    visits = ",".join(str(value) for value in config.get("visits", []))
    visit_indices = config.get("visit_indices")
    hydra_selection = [] if visit_indices is None else [
        f"dataset.visit_indices={json.dumps(visit_indices, separators=(',', ':'))}"
    ]
    common_hydra = [
        f"dataset.root={dataset_root}", f"dataset.split={split}", *hydra_selection
    ]
    backend_commands = {
        "instruction_decomposition": [python, str(fun3du / "run_llm.py"),
                                      *common_hydra, "llm_type=llama_v2"],
        "parent_retrieval": [python, str(fun3du / "run_detection.py"),
                             *common_hydra, "llm_type=llama_v2",
                             f"mask_type={config.get('object_masks', 'owl2_rsam_v2')}"],
        "sparse_grounding": [python, str(fun3du / "run_molmo.py"),
                             *common_hydra, f"exp_root={work_dir}",
                             "exp_name=grounding", "llm_type=llama_v2",
                             f"mask_type={config.get('object_masks', 'owl2_rsam_v2')}",
                             "frame_sampling.mode=score_mean", "frame_sampling.n=50",
                             "molmo_prompt=original"],
        "contracts": [python, str(fun3du / "scripts/contracts/build_structured_contracts.py"),
                      "--data-root", str(dataset_root), "--split", split,
                      "--llm-type", "llama_v2", "--out", str(config["contracts"])],
    }
    processed = Path(config.get(
        "mask3d_processed", fun3du / "exps/prototype_bank/mask3d_f/processed"
    ))
    proposal_commands = [
        [python, str(fun3du / "scripts/bank/mask3d_f/prepare_scenefun3d_mask3d.py"),
         "--data-root", str(dataset_root), "--splits", split,
         "--out-dir", str(processed), "--visits", visits, "--allow-no-gt"],
        [python, str(fun3du / "scripts/bank/mask3d_f/build_sliding_val_crops.py"),
         "--processed", str(processed), "--mode", "validation" if split == "val" else split,
         "--no-require-instance"],
        [mask3d_python, str(fun3du / "scripts/bank/mask3d_f/export_v15_task1.py"),
         "--processed", str(processed),
         "--ckpt", str(Path(config["mask3d_checkpoint"]).expanduser()),
         "--out-dir", str(path_value(config, "proposal_bank")),
         "--visits", visits, "--skip-mask3d-ap", "--skip-eval-task1"],
    ]
    visit_set = set(config.get("visits", [])) or None
    for stage in selected_stages:
        print(f"\n=== {stage} ===", flush=True)
        if stage in backend_commands:
            run_command(backend_commands[stage], fun3du, args.dry_run)
        elif stage == "proposal_bank":
            for command in proposal_commands:
                run_command(command, fun3du, args.dry_run)
        elif args.dry_run:
            print(f"python API: profunc3d.{stage}", flush=True)
        elif stage == "relation_retrieval":
            run_relation_retrieval(config, visit_set)
        elif stage == "instance_identification":
            run_instance_identification(config, visit_set)
        elif stage == "proposal_grounded_association":
            run_pga(config, visit_set)
        elif stage == "prediction":
            run_prediction(config)
    print(f"\nPredictions: {config.get('prediction_dir', work_dir / 'predictions')}", flush=True)


if __name__ == "__main__":
    main()
