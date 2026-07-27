# src/features/engine.py
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..ingestion.registry import ProviderRegistry
from .geo_feature import GeoFeature


class GeoFeatureEngine:
    """
    Core engine responsável por transformar GeoAssets em GeoFeatures.
    """

    def __init__(self) -> None:
        self.providers: dict[str, Any] = {}

    def register_provider(self, name: str, provider: Any) -> None:
        """
        Registra uma instância de provider.
        """
        ProviderRegistry.register(name, type(provider))
        self.providers[name] = provider

    def process(self, config: dict[str, Any]) -> list[GeoFeature]:
        """
        Executa todo o fluxo de ingestão.
        """

        provider_name = config["source"]["provider"]

        if provider_name not in self.providers:
            provider_cls = ProviderRegistry.get(provider_name)
            provider = provider_cls(
                **config["source"].get("config", {})
            )
            self.register_provider(provider_name, provider)
        else:
            provider = self.providers[provider_name]

        if config.get("export"):
            provider.export(**config["export"])

        asset = provider.load_asset(
            Path(config["data"]["root_dir"])
        )

        features = provider.extract_features(asset)

        if isinstance(features, list):
            return features

        return [features]

    # ---------------------------------------------------
    # Compatibilidade com os testes
    # ---------------------------------------------------

    def ingest_and_extract(
        self,
        config: dict[str, Any],
    ) -> list[GeoFeature]:
        """
        Alias utilizado pelos testes de integração.
        """
        return self.process(config)
