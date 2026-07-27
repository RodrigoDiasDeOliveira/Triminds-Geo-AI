import logging
from typing import ClassVar

from .base import GeoProvider

logger = logging.getLogger(__name__)


class ProviderRegistry:
    """Plugin-friendly registry for geospatial providers."""

    _providers: ClassVar[dict[str, type[GeoProvider]]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        provider_class: type[GeoProvider],
    ) -> None:
        """Register a provider implementation."""
        cls._providers[name] = provider_class
        logger.info("Provider registrado: %s", name)

    @classmethod
    def get(
        cls,
        name: str,
    ) -> type[GeoProvider]:
        """Return a registered provider."""
        if name not in cls._providers:
            raise ValueError(
                f"Provider '{name}' não encontrado. Registrados: {list(cls._providers)}"
            )

        return cls._providers[name]

    @classmethod
    def list_providers(cls) -> dict[str, str]:
        """Return registered provider names."""
        return {name: provider.__name__ for name, provider in cls._providers.items()}
