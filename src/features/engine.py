# src/features/engine.py
from pathlib import Path
from typing import Any

from ..ingestion.registry import ProviderRegistry
from .geo_feature import GeoFeature


class GeoFeatureEngine:
    """Coração do Triminds - transforma GeoAssets em GeoFeatures"""

    def __init__(self):
        self.providers: dict[str, Any] = {}

    def register_provider(self, name: str, provider: Any) -> None:
        """Registra um provider no engine"""
        ProviderRegistry.register(name, type(provider))
        self.providers[name] = provider

    def process(self, config: dict) -> list[GeoFeature]:
        """Processa do source até gerar features"""
        provider_name = config["source"]["provider"]

        # Recupera ou cria o provider
        if provider_name not in self.providers:
            provider_class = ProviderRegistry.get(provider_name)
            provider = provider_class(**config["source"].get("config", {}))
            self.register_provider(provider_name, provider)
        else:
            provider = self.providers[provider_name]

        # Export (se necessário)
        if "export" in config:
            _ = provider.export(**config["export"])  # task não usado aqui

        # Carrega asset e extrai features
        asset = provider.load_asset(Path(config["data"]["root_dir"]))
        features = provider.extract_features(asset)

        return [features] if not isinstance(features, list) else features
