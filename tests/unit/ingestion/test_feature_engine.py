import pytest
from src.features.engine import GeoFeatureEngine


def test_geo_feature_engine_initialization():
    engine = GeoFeatureEngine()
    assert engine is not None
    assert len(engine.providers) == 0


def test_register_provider(config):
    engine = GeoFeatureEngine()
    # Aqui você pode mockar um provider para teste
    assert True  # placeholder - expandir depois