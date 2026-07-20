# src/features/engine.py (versão final)
from pathlib import Path
from typing import Dict, List
from ..ingestion.registry import ProviderRegistry
from .geo_feature import GeoFeature
from ..ingestion.base import GeoAsset

class GeoFeatureEngine:
    def __init__(self):
        self.providers = {}
    
    def register_provider(self, name: str, provider):
        ProviderRegistry.register(name, type(provider))
        self.providers[name] = provider
    
    def process(self, config: Dict) -> List[GeoFeature]:
        provider_name = config["source"]["provider"]
        provider = self.providers.get(provider_name)
        
        if not provider:
            provider_class = ProviderRegistry.get(provider_name)
            provider = provider_class(**config["source"].get("config", {}))
            self.register_provider(provider_name, provider)
        
        task = provider.export(**config.get("export", {}))
        asset = provider.load_asset(Path(config["data"]["root_dir"]))
        features = provider.extract_features(asset)
        
        return [features] if not isinstance(features, list) else features