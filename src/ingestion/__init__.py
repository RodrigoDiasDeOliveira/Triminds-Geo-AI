# src/ingestion/__init__.py
from .base import GeoProvider
from .providers.google_embedding import GoogleEmbeddingProvider
from .registry import ProviderRegistry

# Registro automático ao importar o módulo
ProviderRegistry.register("google_embedding", GoogleEmbeddingProvider)

__all__ = ["ProviderRegistry", "GeoProvider", "GoogleEmbeddingProvider"]
