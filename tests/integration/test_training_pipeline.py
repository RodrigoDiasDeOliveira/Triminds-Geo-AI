from unittest.mock import patch

import pytest
import yaml

from src.pipelines.training_pipeline import main


@pytest.fixture
def test_config(tmp_path):
    config = {
        "project": {"name": "Test Triminds"},
        "source": {
            "type": "embedding",
            "provider": "google_embedding",
            "config": {"bucket_name": "test-bucket"}
        },
        "data": {
            "root_dir": str(tmp_path / "embeddings"),
            "year": 2023,
            "dataloader": {"batch_size": 4, "num_workers": 0}
        },
        "model": {
            "name": "resnet50",
            "use_adapter": True,
            "adapter_out_channels": 64,
            "in_channels": 64,
            "num_classes": 5,
            "pretrained": False
        },
        "training": {
            "num_epochs": 1,
            "learning_rate": 0.001
        }
    }
    
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    return config_path


@patch('src.features.engine.GeoFeatureEngine')
@patch('src.pipelines.training_pipeline.GoogleEmbeddingDataset')
def test_training_pipeline_smoke(mock_dataset, mock_engine, test_config, tmp_path):
    """Teste de fumaça - verifica se o pipeline roda sem erros graves"""
    mock_engine.return_value.process.return_value = []
    mock_dataset.return_value.__len__.return_value = 10
    
    # Executa o pipeline
    try:
        main(str(test_config))
        assert True  # Se chegou aqui, passou no teste básico
    except Exception as e:
        pytest.fail(f"Pipeline falhou com erro: {e}")


def test_training_pipeline_with_config(test_config):
    """Verifica se o config é carregado corretamente"""
    config = yaml.safe_load(open(test_config))
    assert config["source"]["provider"] == "google_embedding"
    assert config["model"]["use_adapter"] is True
    assert config["data"]["channels"] == 64