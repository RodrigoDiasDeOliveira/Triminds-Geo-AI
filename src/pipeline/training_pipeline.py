import torch
import yaml

from torch.utils.data import DataLoader
from torch import nn, optim

from src.data_loader.dataset import (
    SatelliteDataset,
    default_transforms
)

from src.models.model_factory import build_model
from src.training.trainer import Trainer


def load_config(path):

    with open(path, "r") as file:
        return yaml.safe_load(file)


def main():

    training_config = load_config("config/training.yaml")
    model_config = load_config("config/model.yaml")

    train_dataset = SatelliteDataset(
        image_paths=[],
        labels=[],
        transform=default_transforms()
    )

    val_dataset = SatelliteDataset(
        image_paths=[],
        labels=[],
        transform=default_transforms()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_config["batch_size"],
        shuffle=False
    )

    model = build_model(
        model_name=model_config["model_name"],
        num_classes=model_config["num_classes"]
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config["learning_rate"]
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    for epoch in range(training_config["epochs"]):

        train_loss = trainer.train_epoch()
        val_loss = trainer.validate()

        print(
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )


if __name__ == "__main__":
    main()