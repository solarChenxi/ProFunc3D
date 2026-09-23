# Fun3DU frontend notice

This directory contains the minimal Fun3DU frontend used by ProFunc3D:
instruction preprocessing, OWLv2/RobustSAM retrieval, Molmo grounding, and
SceneFun3D calibration/data utilities.

- Upstream project: <https://github.com/tev-fbk/fun3du>
- Source revision used during development: `0e124d3624a8d515d972ead086da086a43a0dccf`
- Paper: *Functionality Understanding and Segmentation in 3D Scenes*

The files have been reduced to the runtime dependency closure and adapted only
where necessary to use repository-relative paths and ProFunc3D configuration.
Substantial ProFunc3D-specific changes (structured contracts and Mask3D-F data
export) are identified by their placement under `scripts/contracts/` and
`scripts/bank/mask3d_f/`.

The pinned upstream revision does not include a top-level software LICENSE
file. Before publishing a redistributed copy, obtain/confirm redistribution
permission from the Fun3DU authors or replace this directory with a pinned
submodule. This notice is not a substitute for that permission.
