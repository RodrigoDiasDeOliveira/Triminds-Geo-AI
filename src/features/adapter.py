# src/features/adapter.py
"""
Embedding Adapter
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EmbeddingAdapter(nn.Module):
    """
    Adapter responsável por converter embeddings para o formato esperado
    por modelos pré-treinados.
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 3,
        bottleneck: int = 32,
        bottleneck_dim: int | None = None,
    ) -> None:

        super().__init__()

        if bottleneck_dim is not None:
            bottleneck = bottleneck_dim

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
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.adapter(x)


def get_adapter(
    adapter_type: str = "embedding",
    in_channels: int = 64,
    out_channels: int = 64,
    bottleneck_dim: int = 32,
) -> nn.Module:
    """
    Factory para criação de adapters.
    """

    adapter_type = adapter_type.lower()

    if adapter_type == "identity":
        return nn.Identity()

    if adapter_type == "embedding":
        return EmbeddingAdapter(
            in_channels=in_channels,
            out_channels=out_channels,
            bottleneck_dim=bottleneck_dim,
        )

    raise ValueError(
        f"Unknown adapter type: {adapter_type}"
    )


__all__ = [
    "EmbeddingAdapter",
    "get_adapter",
]
