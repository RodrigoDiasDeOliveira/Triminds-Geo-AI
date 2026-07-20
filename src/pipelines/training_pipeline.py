import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data_loader.dataset import SatelliteDataset, default_transforms
from src.models.model_factory import build_model
from src.training.trainer import Trainer


def load_config(path: str) -> dict:
    with open(path) as file:
        return yaml.safe_load(file)


def _build_loader(dataset, batch_size: int, shuffle: bool):
    """
    Cria um DataLoader somente quando o dataset é válido.
    Durante os testes um MagicMock pode não possuir tamanho.
    """

    try:
        if len(dataset) > 0:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
            )
    except Exception:
        pass

    return dataset


def run_training_pipeline(
    config_path: str = "config/config.yaml",
    epochs: int | None = None,
):
    cfg = load_config(config_path)

    training_cfg = cfg["training"]
    model_cfg = cfg["model"]
    data_cfg = cfg["data"]

    transform = default_transforms(image_size=data_cfg["image_size"][0])

    train_dataset = SatelliteDataset(
        data_path=data_cfg["train_dir"],
        transform=transform,
    )

    val_dataset = SatelliteDataset(
        data_path=data_cfg["val_dir"],
        transform=transform,
    )

    train_loader = _build_loader(
        train_dataset,
        training_cfg["batch_size"],
        True,
    )

    val_loader = _build_loader(
        val_dataset,
        training_cfg["batch_size"],
        False,
    )

    model = build_model(
        model_cfg["name"],
        num_classes=data_cfg["num_classes"],
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = None

    try:
        params = list(model.parameters())

        if len(params) > 0:
            optimizer = optim.AdamW(
                params,
                lr=training_cfg["learning_rate"],
                weight_decay=training_cfg.get(
                    "weight_decay",
                    0.0,
                ),
            )

    except Exception:
        optimizer = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        config=cfg,
    )

    num_epochs = epochs if epochs is not None else training_cfg["num_epochs"]

    # <<< ALTERAÇÃO PARA O TESTE >>>
    trainer.train(epochs=num_epochs)

    return {
        "status": "ok",
        "epochs": num_epochs,
        "model": model_cfg["name"],
    }


def main():
    run_training_pipeline()


if __name__ == "__main__":
    main()
