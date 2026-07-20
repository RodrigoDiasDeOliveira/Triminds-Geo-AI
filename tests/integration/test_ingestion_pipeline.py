"""
Testes de Integração para Ingestion Pipeline
"""

import pytest
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.workflows.ingestion_pipeline import IngestionPipeline


@pytest.fixture
def sample_config(tmp_path):
    config_data = {
        "source": {
            "type": "embedding",
            "provider": "google_embedding",
            "config": {
                "bucket_name": "test-bucket-triminds"
            }
        },
        "data": {
            "root_dir": str(tmp_path / "embeddings"),
            "year": 2023
        },
        "export": {
            "region": [-46.8, -23.7, -46.5, -23.4]
        }
    }
    
    config_path = tmp_path / "ingestion_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    
    return config_path


@patch('src.features.engine.GeoFeatureEngine')
@patch('src.vectorstore.client.VectorStoreClient')
def test_ingestion_pipeline_full_flow(mock_vectorstore, mock_engine, sample_config):
    """Teste completo do pipeline de ingestão"""
    
    # Mock do engine
    mock_feature = MagicMock()
    mock_feature.feature_id = "test_feature_001"
    mock_engine.return_value.ingest_and_extract.return_value = [mock_feature]
    
    # Mock do vector store
    mock_vectorstore.return_value.upsert.return_value = None
    
    # Executa o pipeline
    pipeline = IngestionPipeline()
    result = pipeline.run({
        "source": {"provider": "google_embedding", "config": {"bucket_name": "test"}},
        "data": {"root_dir": "/tmp/test"},
        "export": {}
    })
    
    assert len(result) == 1
    mock_engine.return_value.ingest_and_extract.assert_called_once()
    mock_vectorstore.return_value.upsert.assert_called_once()


def test_ingestion_pipeline_with_real_config(sample_config):
    """Carrega config real e verifica estrutura"""
    with open(sample_config) as f:
        config = yaml.safe_load(f)
    
    assert config["source"]["provider"] == "google_embedding"
    assert "bucket_name" in config["source"]["config"]
    assert config["data"]["year"] == 2023