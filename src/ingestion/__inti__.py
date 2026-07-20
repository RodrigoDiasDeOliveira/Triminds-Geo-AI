# src/ingestion/__init__.py
from .registry import ProviderRegistry
from .providers.google_embedding import GoogleEmbeddingProvider

# Registro automático ao importar o módulo
ProviderRegistry.register("google_embedding", GoogleEmbeddingProvider)

__all__ = ["ProviderRegistry", "GeoProvider", "GoogleEmbeddingProvider"]