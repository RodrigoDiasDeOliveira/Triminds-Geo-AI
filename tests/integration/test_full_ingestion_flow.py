"""
Teste de Integração End-to-End do Fluxo Completo
"""

import pytest

from src.workflows.ingestion_pipeline import IngestionPipeline


@pytest.mark.skip(reason="Temporariamente desabilitado: requer PostgreSQL/VectorStore configurado.")
@pytest.mark.integration
def test_full_ingestion_to_vectorstore_flow():
    """Testa do export até salvar no vector store."""

    pipeline = IngestionPipeline()

    config = {
        "source": {
            "provider": "google_embedding",
            "config": {
                "bucket_name": "test-bucket",
            },
        },
        "data": {
            "root_dir": "/tmp/test_embeddings",
            "year": 2023,
        },
        "export": {
            "region": [
                -46.8,
                -23.7,
                -46.5,
                -23.4,
            ],
        },
    }

    result = pipeline.run(config)

    assert result is not None
