# src/ingestion/registry.py
import logging

from .base import GeoProvider


class ProviderRegistry:
    """Registry plugin-friendly para provedores"""
    
    _providers: dict[str, type[GeoProvider]] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: type[GeoProvider]):
        """Permite registro dinâmico (plugins)"""
        cls._providers[name] = provider_class
        logging.info(f"Provider registrado: {name}")
    
    @classmethod
    def get(cls, name: str) -> type[GeoProvider]:
        if name not in cls._providers:
            raise ValueError(f"Provider '{name}' não encontrado. Registrados: {list(cls._providers.keys())}")
        return cls._providers[name]
    
    @classmethod
    def list_providers(cls) -> dict:
        return {name: cls._providers[name].__name__ for name in cls._providers}