# Released artifacts

Place the downloadable validation artifacts under `assets/paper_checkpoint/`:

```text
paper_checkpoint/
├── contracts.json
├── grounding_observations/
├── instance_timelines/
├── selected_instances/
├── parent_retrieval_overrides.json
└── proposal_bank/
    ├── merged/
    └── xyz_cache/
```

These files contain model predictions and frozen proposal indices only. They do
not contain SceneFun3D ground-truth masks, IoUs, or oracle selections. The raw
SceneFun3D assets must be downloaded from the dataset provider separately.

