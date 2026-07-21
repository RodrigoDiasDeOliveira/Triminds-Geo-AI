# src/features/adapter.py
"""
Embedding Adapter

Adaptador para permitir que modelos pré-treinados recebam embeddings
ou entradas com número diferente de canais.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EmbeddingAdapter(nn.Module):
    """
    Adapter para preservar conhecimento pré-treinado enquanto adapta
    o número de canais da entrada.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 3,
        bottleneck: int = 32,
    ) -> None:
        super().__init__()

        self.adapter = nn.Sequential(
            nn.Conv2d(
                in_channels,
                bottleneck,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(bottleneck),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                bottleneck,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Inicializa os pesos do adapter."""

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executa o forward do adapter."""

        return self.adapter(x)


def get_adapter(
    adapter_type: str = "embedding",
    in_channels: int = 64,
    out_channels: int = 64,
    bottleneck_dim: int = 32,
) -> nn.Module:
    """
    Factory para criação de adapters.

    Parameters
    ----------
    adapter_type:
        Tipo do adapter.
    in_channels:
        Número de canais de entrada.
    out_channels:
        Número de canais de saída.
    bottleneck_dim:
        Dimensão intermediária.

    Returns
    -------
    nn.Module
        Adapter configurado.
    """

    adapter_type = adapter_type.lower()

    if adapter_type == "embedding":
        return EmbeddingAdapter(
            in_channels=in_channels,
            out_channels=out_channels,
            bottleneck=bottleneck_dim,
        )

    raise ValueError(f"Unknown adapter type: {adapter_type}")


__all__ = [
    "EmbeddingAdapter",
    "get_adapter",
]
