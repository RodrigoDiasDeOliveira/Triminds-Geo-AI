# src/pipelines/training_pipeline.py
from __future__ import annotations

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
    """
    Load the training configuration from a YAML file.
    """
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def _build_loader(
    dataset: Dataset | Any,
    batch_size: int,
    shuffle: bool,
) -> DataLoader | Dataset:
    """
    Build a DataLoader when the dataset supports batching.

    Some unit tests inject mock datasets that are not fully compatible with
    PyTorch's DataLoader. In those cases the dataset itself is returned.
    """

    if not isinstance(dataset, Sized):
        return dataset

    if len(dataset) == 0:
        return dataset

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def run_training_pipeline(
    config_path: str = "config/config.yaml",
    epochs: int | None = None,
) -> dict[str, str | int]:
    """
    Execute the model training pipeline.
    """

    cfg = load_config(config_path)

    training_cfg = cfg["training"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    # ==========================================================
    # Compatibility between legacy and new configuration formats
    # ==========================================================

    image_size = data_cfg.get(
        "image_size",
        [224, 224],
    )[0]

    train_dir = data_cfg.get(
        "train_dir",
        data_cfg.get("root_dir"),
    )

    val_dir = data_cfg.get(
        "val_dir",
        train_dir,
    )

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

    transform = default_transforms(
        image_size=image_size,
    )

    train_dataset = GoogleEmbeddingDataset(
        data_path=train_dir,
        transform=transform,
    )

    val_dataset = GoogleEmbeddingDataset(
        data_path=val_dir,
        transform=transform,
    )

    train_loader = _build_loader(
        train_dataset,
        batch_size,
        True,
    )

    val_loader = _build_loader(
        val_dataset,
        batch_size,
        False,
    )

    model = build_model(
        model_cfg["name"],
        num_classes=num_classes,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg.get(
            "weight_decay",
            0.0,
        ),
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
        config=cfg,
    )

    trainer.train(
        epochs=epochs or training_cfg["num_epochs"],
    )

    return {
        "status": "ok",
        "epochs": epochs or training_cfg["num_epochs"],
        "model": model_cfg["name"],
    }


def main(
    config_path: str = "config/config.yaml",
) -> dict[str, str | int]:
    """
    Entry point for the training pipeline.
    """
    return run_training_pipeline(config_path)


if __name__ == "__main__":
    main()
