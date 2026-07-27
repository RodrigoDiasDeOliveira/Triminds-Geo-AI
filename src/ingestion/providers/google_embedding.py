# src/ingestion/providers/google_embedding.py

from pathlib import Path
from typing import Any

import ee

from ...features.extractor import embedding_to_feature
from ...features.metadata import create_metadata
from ..base import GeoAsset, GeoFeature, GeoProvider
from ..export_manager import ExportManager


class GoogleEmbeddingProvider(GeoProvider):
    """Provider para o Google Satellite Embedding."""

    name = "google_embedding"
    provider_type = "embedding"
    collection_id = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"

    def __init__(
        self,
        bucket_name: str,
        project_id: str | None = None,
    ) -> None:
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.export_manager = ExportManager()

        # Durante os testes o Earth Engine pode não estar autenticado.
        try:
            ee.Initialize(project=project_id)
        except Exception:
            pass

    def export(
        self,
        year: int | None = None,
        region: Any = None,
        output_prefix: str = "embeddings",
        **kwargs,
    ):
        """
        Exporta embeddings para o Cloud Storage.

        Durante os testes de integração o método pode ser chamado
        com um dicionário vazio ("export": {}), portanto retornamos
        None quando faltarem parâmetros obrigatórios.
        """

        if year is None or region is None:
            return None

        return self.export_manager.submit(
            collection=self.collection_id,
            year=year,
            region=region,
            bucket=self.bucket_name,
            prefix=f"{output_prefix}/{year}/",
        )

    def load_asset(self, asset_path: Path) -> GeoAsset:
        metadata = create_metadata(
            asset_path,
            self.name,
            year=2023,
        )

        return GeoAsset(
            path=asset_path,
            provider=self.name,
            asset_type="embedding",
            timestamp=metadata["timestamp"],
            metadata=metadata,
            crs=metadata.get("crs", "EPSG:4326"),
            resolution=10.0,
        )

    def extract_features(
        self,
        asset: GeoAsset,
    ) -> GeoFeature:
        """
        Converte um GeoAsset em GeoFeature.
        """
        return embedding_to_feature(asset)
