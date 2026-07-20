from src.features.geo_feature import GeoFeature
import torch


def test_geo_feature_creation():
    feature = GeoFeature(
        geometry=None,
        timestamp="2023-01-01",
        embedding=torch.randn(64),
        source="google_embedding",
        confidence=0.95
    )
    
    assert feature.source == "google_embedding"
    assert feature.confidence == 0.95
    assert feature.feature_id is not None


def test_geo_feature_to_dict():
    feature = GeoFeature(
        geometry=None,
        timestamp="2023-01-01",
        embedding=torch.randn(64),
        source="google"
    )
    data = feature.to_dict()
    assert "id" in data
    assert "source" in data
    assert "timestamp" in data