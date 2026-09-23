# Third-party backends

ProFunc3D carries the complete runtime subset of the Fun3DU frontend under
`Fun3DU/`. It includes:

- `run_llm.py`: LLaMA-3.1 instruction decomposition;
- `run_detection.py`: OWLv2 + RobustSAM object retrieval;
- `run_molmo.py`: Molmo sparse point grounding on at most 50 frames;
- `utils/sun3d/`: SceneFun3D RGB-D, pose, and calibration readers;
- `scripts/contracts/`: structured relational contract construction;
- `scripts/bank/mask3d_f/`: Mask3D-F preprocessing, sliding crops, inference,
  proposal merging, and Task-1 export/evaluation.

The heavyweight generic Mask3D framework is kept as the pinned
`third_party/Mask3D` Git submodule instead of duplicating checkpoints, logs, or
training outputs. The setup helper supports both a Git submodule checkout and
a source archive without submodule metadata:

```bash
bash scripts/setup_mask3d.sh
```

The expected Mask3D revision is:

```text
https://github.com/JonasSchult/Mask3D.git
11bd5ff94477ff7194e9a7c52e9fae54d73ac3b5
```

Mask3D retains its upstream MIT license. See `Fun3DU/NOTICE.md` for Fun3DU
provenance and the redistribution caveat that must be resolved before making
the repository public.
