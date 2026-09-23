# Mask3D-F proposal generation

These are the complete ProFunc3D-specific scripts that turn SceneFun3D point
clouds into the frozen 3D proposal bank used by Proposal-grounded Association.
They are called automatically by `profunc3d.pipeline`, but can also be run
separately.

From the `ProFunc3D` repository root:

```bash
# 1. Convert SceneFun3D scans to Mask3D format.
python third_party/Fun3DU/scripts/bank/mask3d_f/prepare_scenefun3d_mask3d.py \
  --data-root data/scenefun3d --splits val \
  --out-dir outputs/val/mask3d_processed --allow-no-gt

# 2. Build 4 m sliding windows with 2 m stride.
python third_party/Fun3DU/scripts/bank/mask3d_f/build_sliding_val_crops.py \
  --processed outputs/val/mask3d_processed \
  --mode validation --no-require-instance

# 3. Run the trained Mask3D-F checkpoint and merge window proposals.
/path/to/mask3d/python \
  third_party/Fun3DU/scripts/bank/mask3d_f/export_v15_task1.py \
  --processed outputs/val/mask3d_processed \
  --ckpt checkpoints/mask3d_f.ckpt \
  --out-dir outputs/val/proposal_bank \
  --skip-mask3d-ap --skip-eval-task1
```

The proposal bank consumed by ProFunc3D has this layout:

```text
outputs/val/proposal_bank/
├── merged/<visit>.npz
└── <visit>/task1_input.npz
```

Each `merged/<visit>.npz` stores sparse point-index masks using
`idx_concat`/`idx_offsets`, plus `pred_scores` and `pred_classes`. No ground
truth is read during normal inference. AP evaluation is optional and only
enabled explicitly on annotated validation data.

The generic Mask3D framework is provided through the repository's pinned
`third_party/Mask3D` submodule. Model checkpoints and generated data are not
committed.
