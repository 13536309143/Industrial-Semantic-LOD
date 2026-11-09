"""Strict loading for the industrial classifier: reject partial/random weights."""

from pathlib import Path

import torch


def load_classifier_checkpoint(model, checkpoint_path: Path):
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state = checkpoint
    for key in ("model", "net", "network", "state_dict", "base_model"):
        if key in checkpoint:
            state = checkpoint[key]
            break
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    print(f"Loaded complete PointNeXt checkpoint: {checkpoint_path} (epoch={checkpoint.get('epoch', '?')})")
    return checkpoint
