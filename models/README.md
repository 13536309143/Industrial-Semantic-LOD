# Local model weights

Model weights are stored locally and ignored by Git. Source configuration, class ordering, and selected training evidence are tracked separately.

| Path | Purpose |
| :--- | :--- |
| `pointnext/industrial-part-14class-best.pth` | Default supervised classifier for scene analysis; 14 classes, 8,192 XYZ points |
| `.artifacts/training/pointnext/industrial-part-14class/industrial-part-14class-latest.pth` (repository root) | Preserved previous run's latest checkpoint for resuming training |

Train a compatible classifier or supply the existing industrial checkpoint, then copy the best file to the default path. Do not rename a generic ModelNet40 checkpoint to this name: its class head and labels are incompatible.

Alternatively, set `POINTNEXT_CKPT` to an absolute checkpoint path before running either analysis batch. The class index order must match `tools/pointnext_lod/industrial_part_14classes.txt`.

PointCLIP V2 uses pretrained CLIP weights, normally cached by the CLIP loader outside this directory. The default analysis uses `ViT-B/16`; its first load may download weights.

[Complete workflow](../docs/guides/end-to-end.md) · [完整中文流程](../docs/guides/end-to-end.zh-CN.md)
