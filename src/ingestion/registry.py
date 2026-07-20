# src/ingestion/registry.py
from typing import Dict, Type, Optional
from .base import GeoProvider
import logging

class ProviderRegistry:
    """Registry plugin-friendly para provedores"""
    
    _providers: Dict[str, Type[GeoProvider]] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: Type[GeoProvider]):
        """Permite registro dinâmico (plugins)"""
        cls._providers[name] = provider_class
        logging.info(f"Provider registrado: {name}")
    
    @classmethod
    def get(cls, name: str) -> Type[GeoProvider]:
        if name not in cls._providers:
            raise ValueError(f"Provider '{name}' não encontrado. Registrados: {list(cls._providers.keys())}")
        return cls._providers[name]
    
    @classmethod
    def list_providers(cls) -> Dict:
        return {name: cls._providers[name].__name__ for name in cls._providers}