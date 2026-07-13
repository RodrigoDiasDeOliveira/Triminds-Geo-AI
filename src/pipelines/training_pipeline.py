import torch
import yaml
from torch.utils.data import DataLoader
from torch import nn, optim

from src.data_loader.dataset import SatelliteDataset, default_transforms
from src.models.model_factory import build_model
from src.training.trainer import Trainer


def load_config(path: str) -> dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)


def main():
    cfg = load_config("config/config.yaml")
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    transform = default_transforms(image_size=data_cfg["image_size"][0])

    train_dataset = SatelliteDataset(data_path=data_cfg["train_dir"], transform=transform)
    val_dataset = SatelliteDataset(data_path=data_cfg["val_dir"], transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=training_cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_cfg["batch_size"], shuffle=False)

    model = build_model(model_cfg["name"], num_classes=data_cfg["num_classes"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=training_cfg["learning_rate"],
                            weight_decay=training_cfg.get("weight_decay", 0.0))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device)
    trainer.train(epochs=training_cfg["num_epochs"])


if __name__ == "__main__":
    main()
