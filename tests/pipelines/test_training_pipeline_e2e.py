from unittest.mock import MagicMock, patch

import torch.nn as nn

from src.pipelines.training_pipeline import run_training_pipeline


@patch("src.pipelines.training_pipeline.Trainer")
@patch("src.pipelines.training_pipeline.build_model")
@patch("src.pipelines.training_pipeline.SatelliteDataset")
@patch("src.pipelines.training_pipeline.load_config")
def test_pipeline_execution(
    mock_cfg,
    mock_dataset,
    mock_build_model,
    mock_trainer,
):
    mock_cfg.return_value = {
        "training": {
            "batch_size": 2,
            "learning_rate": 1e-3,
            "num_epochs": 1,
        },
        "model": {
            "name": "resnet50",
        },
        "data": {
            "train_dir": "train",
            "val_dir": "val",
            "image_size": [64, 64],
            "num_classes": 10,
        },
    }

    # Mock do Dataset
    mock_dataset.return_value = MagicMock()

    # Modelo PyTorch REAL
    mock_build_model.return_value = nn.Linear(10, 10)

    trainer_instance = MagicMock()
    mock_trainer.return_value = trainer_instance

    result = run_training_pipeline(epochs=1)

    trainer_instance.train.assert_called_once_with(epochs=1)

    assert result == {
        "status": "ok",
        "epochs": 1,
        "model": "resnet50",
    }