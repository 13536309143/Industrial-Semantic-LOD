# Recorded industrial part classification run

This folder retains compact evidence for the recorded 14-class PointNeXt run.

- `training.log`: original console log, including 100 epochs and best-checkpoint evaluation.
- `run-config.yaml`: original serialized runtime configuration for provenance. It contains Python-specific YAML tags and historical absolute paths; it is not a portable training recipe.
- The portable recipe remains under `tools/pointnext_lod/PointNeXt/cfgs/industrial_part/pointnext-s-14class-randomrot-8192-fast.yaml`.
- Best weights are local under `models/pointnext/`; latest weights, trainer logs, and TensorBoard events are retained under `.artifacts/training/pointnext/industrial-part-14class/`.

Recorded result: epoch 85, OA 96.19%, mAcc 93.86%, seed 3098. Validation and test share the same `test` split, so this is a development evaluation rather than an independent test result.

The archive layout organizes an existing run; it does not claim a new training experiment.
