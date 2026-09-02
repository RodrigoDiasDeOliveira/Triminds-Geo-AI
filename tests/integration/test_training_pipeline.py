from pathlib import Path

import pytest
import yaml


@pytest.fixture
def test_config(tmp_path):
    config = {
        "project": {"name": "Test Triminds"},
        "source": {"type": "raster", "provider": "local", "config": {}},
        "data": {
            "root_dir": str(tmp_path / "demo"),
            "channels": 3,
            "image_size": [224, 224],
            "num_classes": 10,
            "dataloader": {"batch_size": 2, "num_workers": 0},
        },
        "model": {
            "name": "resnet50",
            "use_adapter": False,
            "adapter_out_channels": 64,
            "in_channels": 3,
            "num_classes": 10,
            "pretrained": False,
        },
        "training": {"num_epochs": 1, "learning_rate": 0.001},
    }

    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file)

    return config_path


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


def test_demo_config_fixture(test_config):
    with open(test_config, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    assert config["source"]["provider"] == "local"
    assert config["model"]["in_channels"] == 3
    assert config["model"]["use_adapter"] is False
    assert config["data"]["channels"] == 3
