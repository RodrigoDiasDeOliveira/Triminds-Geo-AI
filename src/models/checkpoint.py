from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    model_name: str,
    num_classes: int,
    in_channels: int,
    use_adapter: bool,
    adapter_out_channels: int,
    config: dict[str, Any] | None = None,
) -> None:
    """Save a model checkpoint using the project-wide checkpoint contract."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "num_classes": num_classes,
        "in_channels": in_channels,
        "use_adapter": use_adapter,
        "adapter_out_channels": adapter_out_channels,
        "config": config or {},
    }

    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    *,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a project checkpoint into an already-created model."""
    checkpoint_path = Path(path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=True,
    )

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Invalid checkpoint format: {checkpoint_path}. "
            "Expected 'model_state_dict'."
        )

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    return checkpoint
