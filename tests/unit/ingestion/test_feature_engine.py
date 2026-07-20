from src.features.engine import GeoFeatureEngine


def test_geo_feature_engine_initialization():
    engine = GeoFeatureEngine()
    assert engine is not None
    assert isinstance(engine.providers, dict)
    assert len(engine.providers) == 0


def test_register_provider():
    engine = GeoFeatureEngine()
    
    from src.ingestion.providers.google_embedding import GoogleEmbeddingProvider
    provider = GoogleEmbeddingProvider(bucket_name="test-bucket")
    
    engine.register_provider("google_embedding", provider)
    
    assert "google_embedding" in engine.providers
    assert engine.providers["google_embedding"] == provider