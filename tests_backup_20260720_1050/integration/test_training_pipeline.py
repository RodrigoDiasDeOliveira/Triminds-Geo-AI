from unittest.mock import patch

from src.pipelines.training_pipeline import run_training_pipeline


@patch("src.pipelines.training_pipeline.Trainer")
@patch("src.pipelines.training_pipeline.build_model")
@patch("src.pipelines.training_pipeline.SatelliteDataset")
@patch("src.pipelines.training_pipeline.load_config")
def test_pipeline_execution(mock_cfg, mock_ds, mock_build, mock_trainer):
    mock_cfg.return_value = {
        "training": {"batch_size": 2, "learning_rate": 1e-3, "num_epochs": 1},
        "model": {"name": "resnet50"},
        "data": {"train_dir": "x", "val_dir": "y", "image_size": [64, 64], "num_classes": 10},
    }
    mock_ds.return_value = [(0, 0)]
    result = run_training_pipeline(epochs=1)
    assert result is not None
