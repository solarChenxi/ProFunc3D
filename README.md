# ProFunc3D

Official implementation of **ProFunc3D: Bridging 2D Grounding and 3D Proposals for 3D Functionality Grounding**.

ProFunc3D predicts a functional 3D region from a natural-language instruction, an RGB-D scanning video, and its registered point cloud. It contains the two components named in the paper:

1. **Proposal-grounded Association (PGA)** directly associates sparse 2D grounding points from the original video with a frozen bank of coherent 3D functional proposals and aggregates proposal evidence across views.
2. **Instance-aware Reweighting (IAR)** identifies the physical parent instance referred to by a relational instruction and reweights the same grounding observations, suppressing evidence from distractor instances without introducing additional grounding frames.

On the SceneFun3D Task-2 validation set, the released configuration obtains:

| mAP | AP50 | AP25 | mAR | AR50 | AR25 | mIoU |
|---:|---:|---:|---:|---:|---:|---:|
| 23.124 | 31.910 | 43.146 | 28.607 | 36.180 | 44.045 | 25.182 |

The machine-readable reference values are provided in
[`assets/expected_metrics.json`](assets/expected_metrics.json).

## Method

Given instruction `q`, scanning video `V`, and point cloud `P`, a frozen vision-language model predicts a sparse point `p_t` in each of at most 50 retrieved frames. A language-agnostic proposal bank contains coherent 3D regions `Π={π_i}`.

### Proposal-grounded Association (PGA)

For each grounding point, PGA projects the visible points of every proposal into the source frame, measures the image-space point-to-proposal distance, and assigns the observation to one proposal. Local surface support within `r_s=20 px` resolves overlapping proposals. The resulting assignments are aggregated into generic evidence `V_i^g`; non-relational instructions select the proposal with maximum generic evidence.

The released checkpoint uses a deterministic overlap condition: local surface support resolves proposals whose nearest projected distances are numerically tied.

### Instance-aware Reweighting (IAR)

For a relational instruction, LLaMA-3.1 decomposes the instruction into a parent object, relation, and reference object. OWLv2+RobustSAM parent observations are lifted to 3D and clustered with DBSCAN (`0.75 m`) into persistent physical-instance tracks. The track most consistent with the relation is selected.

The selected track is aligned with the original grounding frames within `±0.9 s`; no extra Molmo frames are introduced. For aligned frame `t`,

```text
q_t = detection_confidence_t × frame_completeness_t
q_bar_t = q_t / max(q)
w_t = sigmoid((q_bar_t - beta) / gamma)
```

with `beta=0.20` and `gamma=0.02`. Unaligned observations receive zero weight. If no grounding observation aligns with the selected track, ProFunc3D falls back to unweighted PGA, as described in the paper.

## Repository layout

```text
ProFunc3D/
├── configs/
│   ├── profunc3d.yaml          # fresh end-to-end inference
│   └── reproduce_paper.yaml    # exact frozen validation replay
├── profunc3d/
│   ├── pga.py                  # Eqs. (2)-(5)
│   ├── iar.py                  # Eqs. (8)-(12) and temporal alignment
│   ├── tracks.py               # physical instance identification, Eqs. (6)-(7)
│   ├── relation_retrieval.py   # parent/reference OWLv2+RobustSAM retrieval
│   ├── associate.py            # calibrated 2D-point to 3D-proposal association
│   ├── predict.py              # weighted aggregation and final mask export
│   ├── backend.py              # SceneFun3D/Fun3DU data and calibration adapter
│   └── pipeline.py             # end-to-end stage orchestration
├── scripts/                    # explicit CLI entry points
├── tests/                      # unit tests for PGA/IAR behavior
├── third_party/Fun3DU/         # vendored frozen frontend + Mask3D-F scripts
├── third_party/Mask3D/         # pinned upstream Mask3D Git submodule
└── pyproject.toml
```

The implementation, configuration, intermediate artifacts, and command-line interface use the terminology and symbols of the paper.

## Input data

The semantic inputs are:

- a natural-language instruction;
- RGB frames from an indoor scan;
- the corresponding colored point cloud.

Calibrated 2D-to-3D association additionally requires the aligned depth frames, camera intrinsics, camera poses, timestamps, and point-cloud crop mask. For SceneFun3D, download:

```text
laser_scan_5mm, crop_mask, descriptions,
hires_wide, hires_depth, hires_wide_intrinsics, hires_poses
```

An instruction must be registered in the visit's description file and have a `desc_id`. RGB images without depth and camera calibration are insufficient for geometric 2D-to-3D association.

## Models and dependencies

The paper uses frozen models throughout:

- LLaMA-3.1-8B for instruction decomposition;
- OWLv2 + RobustSAM for parent/reference retrieval;
- Molmo-7B-D for sparse 2D grounding;
- Mask3D-F for the offline proposal bank.

ProFunc3D includes the required Fun3DU frontend and SceneFun3D data backend in
`third_party/Fun3DU`; a separate Fun3DU checkout is not needed. The vendored
runtime subset contains `run_llm.py`, `run_detection.py`, `run_molmo.py`, and
all required `utils/sun3d/` calibration utilities. It also contains the full
Mask3D-F preprocessing, crop-building, inference, merging, and export scripts.

Initialize the generic Mask3D framework and apply the included Mask3D-F patch
once after cloning:

```bash
bash scripts/setup_mask3d.sh
```

Install ProFunc3D and its bundled frontend dependencies:

```bash
cd ProFunc3D
pip install -e '.[frontend]'
```

The Mask3D-F proposal stage uses its own environment. Set `mask3d_python` and
`mask3d_checkpoint` in the configuration. Checkpoints, datasets, generated
proposal banks, and experiment logs are intentionally not bundled.

For LLaMA preprocessing, start Ollama first:

```bash
ollama serve
ollama pull llama3.1
```

## End-to-end inference

Edit [configs/profunc3d.yaml](configs/profunc3d.yaml), then run:

```bash
profunc3d --config configs/profunc3d.yaml
```

The complete stage sequence is:

```text
instruction_decomposition
→ parent_retrieval
→ sparse_grounding (≤50 original frames)
→ contracts
→ relation_retrieval
→ proposal_bank
→ instance_identification
→ proposal_grounded_association
→ prediction
```

To inspect commands without running models:

```bash
profunc3d --config configs/profunc3d.yaml --dry-run
```

To reuse existing frontend and proposal caches:

```bash
profunc3d --config configs/profunc3d.yaml \
  --stages instance_identification,proposal_grounded_association,prediction
```

Individual components are also exposed:

```bash
python -m profunc3d.relation_retrieval --config configs/profunc3d.yaml
python -m profunc3d.tracks             --config configs/profunc3d.yaml
python -m profunc3d.associate          --config configs/profunc3d.yaml
python -m profunc3d.predict            --config configs/profunc3d.yaml
```

## Outputs

Each query produces:

```text
predictions/<visit>_<desc_id>/
├── prediction.npz
├── prediction.json
├── predicted_region.ply
└── scene_with_prediction.ply
```

`prediction.npz` contains:

- `point_mask`: full-resolution binary mask in the input cropped point-cloud order;
- `selected_indices`: selected point indices;
- `selected_proposal`: final proposal ID.

`prediction.json` records the PGA evidence, IAR weights, temporal alignment, selected route, and all inference parameters. `scene_with_prediction.ply` downsamples only background points for visualization; `point_mask` and `predicted_region.ply` retain the complete prediction.

## Reproducing the paper checkpoint

The local OWLv2 cache was extended after the frozen experiment. `configs/reproduce_paper.yaml` therefore supports a validation-only compatibility path: four saved identity observations recover the same prediction-only physical track in the extended timeline. This compatibility step never reads GT, IoU, or an oracle track. It is not used for fresh inference.

With released caches available, run:

```bash
profunc3d --config configs/reproduce_paper.yaml \
  --stages proposal_grounded_association,prediction
```

Fresh inference must use `configs/profunc3d.yaml` and the current `selected_track_detections`; do not set `frozen_track_dir`.

## Tests

```bash
pytest -q
python -m py_compile profunc3d/*.py scripts/*.py
```

The prediction path does not read SceneFun3D annotations, ground-truth masks, oracle proposals, or IoU values.

## 中文说明

这是论文方法的独立开源目录，与原实验工程平行。公开模块名严格对应论文中的 **PGA** 和 **IAR**。完整入口接收带标定的 RGB-D 扫描、点云和自然语言指令，最终输出与输入点云逐点对齐的二值预测区域。默认配置不增加 Molmo 帧：IAR 只给原始至多 50 帧的 grounding evidence 重新赋权。
