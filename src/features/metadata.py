"""
Metadata utilities.

Criação e gerenciamento de metadados para assets geoespaciais.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def create_metadata(
    asset_path: Path,
    provider: str,
    year: int | None = None,
) -> dict[str, Any]:
    """
    Cria um dicionário de metadados para um asset.

    Parameters
    ----------
    asset_path:
        Caminho do asset.
    provider:
        Nome do provider.
    year:
        Ano de referência.

    Returns
    -------
    dict
        Metadados do asset.
    """

    return {
        "filename": asset_path.name,
        "provider": provider,
        "year": year,
        "timestamp": datetime.now(UTC).isoformat(),
        "crs": "EPSG:4326",
    }


__all__ = [
    "create_metadata",
]
