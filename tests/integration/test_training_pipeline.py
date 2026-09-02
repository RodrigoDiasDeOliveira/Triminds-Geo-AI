from pathlib import Path

import torch
import yaml
from PIL import Image

from src.models.checkpoint import load_checkpoint
from src.models.model_factory import build_model
from src.pipelines.training_pipeline import main


def _create_demo_dataset(root: Path) -> None:
    for class_index in range(2):
        class_dir = root / f"class_{class_index}"
        class_dir.mkdir(parents=True)
        for image_index in range(2):
            image = Image.new(
                "RGB",
                (64, 64),
                color=(class_index * 100, image_index * 50, 32),
            )
            image.save(class_dir / f"sample_{image_index}.png")


def _create_demo_config(tmp_path: Path, dataset_root: Path) -> Path:
    config = {
        "project": {"name": "Triminds Geo AI Demo Test"},
        "source": {"type": "raster", "provider": "local", "config": {}},
        "data": {
            "root_dir": str(dataset_root),
            "image_size": [64, 64],
            "channels": 3,
            "num_classes": 2,
            "dataloader": {"batch_size": 2, "num_workers": 0},
        },
        "model": {
            "name": "resnet50",
            "use_adapter": False,
            "adapter_out_channels": 64,
            "in_channels": 3,
            "num_classes": 2,
            "pretrained": False,
        },
        "training": {
            "num_epochs": 1,
            "learning_rate": 0.001,
            "early_stopping_patience": 1,
        },
        "paths": {"artifacts_dir": str(tmp_path / "artifacts")},
        "logging": {"use_mlflow": False},
    }

    config_path = tmp_path / "demo_test.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)
    return config_path


def test_demo_training_produces_loadable_checkpoint(tmp_path):
    """Validate dataset -> training -> checkpoint -> model reload."""
    dataset_root = tmp_path / "dataset"
    _create_demo_dataset(dataset_root)
    config_path = _create_demo_config(tmp_path, dataset_root)

    result = main(str(config_path))

    assert result["status"] == "ok"
    checkpoint_path = tmp_path / "artifacts" / "best_model.pth"
    assert checkpoint_path.exists()

    model = build_model(
        "resnet50",
        num_classes=2,
        pretrained=False,
        in_channels=3,
        use_adapter=False,
    )
    checkpoint = load_checkpoint(checkpoint_path, model, strict=True)

    assert checkpoint["model_name"] == "resnet50"
    assert checkpoint["num_classes"] == 2
    assert checkpoint["in_channels"] == 3
    assert checkpoint["use_adapter"] is False

    model.eval()
    with torch.no_grad():
        prediction = model(torch.randn(1, 3, 64, 64))

    assert prediction.shape == (1, 2)


def test_demo_config_matches_rgb_pipeline():
    config_path = Path("config/demo.yaml")
    assert config_path.exists()

    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert config["source"]["provider"] == "local"
    assert config["data"]["channels"] == 3
    assert config["model"]["name"] == "resnet50"
    assert config["model"]["in_channels"] == 3
    assert config["model"]["use_adapter"] is False


def test_embedding_config_remains_explicit():
    config_path = Path("config/config.yaml")
    assert config_path.exists()

    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert config["source"]["provider"] == "google_embedding"
    assert config["data"]["channels"] == 64
    assert config["model"]["in_channels"] == 64
    assert config["model"]["use_adapter"] is True
    assert config["model"]["name"] == "resnet50"
