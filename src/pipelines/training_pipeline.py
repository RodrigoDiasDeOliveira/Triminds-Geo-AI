# src/pipelines/training_pipeline.py
from __future__ import annotations

import argparse
from collections.abc import Sized
from typing import Any

import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from src.data_loader.dataset import (
    SatelliteDataset,
    default_transforms,
)
from src.models.model_factory import build_model
from src.training.trainer import Trainer

# Alias esperado pelos testes
GoogleEmbeddingDataset = SatelliteDataset


def load_config(path: str) -> dict[str, Any]:
    """Load the training configuration from a YAML file."""
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def _build_loader(
    dataset: Dataset | Any,
    batch_size: int,
    shuffle: bool,
) -> DataLoader | Dataset:
    """Build a DataLoader when the dataset supports batching."""
    if not isinstance(dataset, Sized):
        return dataset

    if len(dataset) == 0:
        return dataset

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def _validate_dataset(dataset: Dataset | Any, split: str) -> None:
    """Fail early when a configured training split contains no samples."""
    if isinstance(dataset, Sized) and len(dataset) == 0:
        raise ValueError(
            f"Dataset split '{split}' is empty. "
            "Check the configured data directory and class folders."
        )


def run_training_pipeline(
    config_path: str = "config/config.yaml",
    epochs: int | None = None,
) -> dict[str, str | int]:
    """Execute the model training pipeline."""
    cfg = load_config(config_path)

    training_cfg = cfg["training"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    image_size = data_cfg.get("image_size", [224, 224])[0]

    train_dir = data_cfg.get(
        "train_dir",
        data_cfg.get("root_dir"),
    )
    val_dir = data_cfg.get("val_dir", train_dir)

    num_classes = data_cfg.get(
        "num_classes",
        model_cfg.get("num_classes", 2),
    )

    batch_size = data_cfg.get(
        "dataloader",
        {},
    ).get(
        "batch_size",
        training_cfg.get("batch_size", 32),
    )

    transform = default_transforms(image_size=image_size)

    train_dataset = GoogleEmbeddingDataset(
        data_path=train_dir,
        transform=transform,
    )
    val_dataset = GoogleEmbeddingDataset(
        data_path=val_dir,
        transform=transform,
    )

    _validate_dataset(train_dataset, "train")
    _validate_dataset(val_dataset, "val")

    train_loader = _build_loader(train_dataset, batch_size, True)
    val_loader = _build_loader(val_dataset, batch_size, False)

    model = build_model(
        model_cfg["name"],
        num_classes=num_classes,
        pretrained=model_cfg.get("pretrained", False),
        in_channels=model_cfg.get("in_channels", 3),
        use_adapter=model_cfg.get("use_adapter", False),
        adapter_out_channels=model_cfg.get("adapter_out_channels", 64),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg.get("weight_decay", 0.0),
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        config=cfg,
    )

    effective_epochs = epochs or training_cfg["num_epochs"]
    trainer.train(epochs=effective_epochs)

    return {
        "status": "ok",
        "epochs": effective_epochs,
        "model": model_cfg["name"],
    }


def main(
    config_path: str = "config/config.yaml",
) -> dict[str, str | int]:
    """Entry point for the training pipeline."""
    return run_training_pipeline(config_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Triminds Geo AI training pipeline.")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to the YAML training configuration.",
    )
    args = parser.parse_args()
    main(args.config)
