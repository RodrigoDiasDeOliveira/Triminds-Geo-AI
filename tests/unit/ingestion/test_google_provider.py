import pytest
from pathlib import Path
from src.ingestion.providers.google_embedding import GoogleEmbeddingProvider


def test_google_provider_initialization():
    provider = GoogleEmbeddingProvider(bucket_name="test-bucket")
    assert provider.name == "google_embedding"
    assert provider.provider_type == "embedding"
    assert provider.collection_id == "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"


def test_provider_has_required_methods():
    provider = GoogleEmbeddingProvider(bucket_name="test-bucket")
    assert hasattr(provider, "export")
    assert hasattr(provider, "load_asset")
    assert hasattr(provider, "extract_features")