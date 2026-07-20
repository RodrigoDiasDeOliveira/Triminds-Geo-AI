import torch

from src.features.adapter import EmbeddingAdapter, get_adapter


def test_embedding_adapter_shape():
    adapter = EmbeddingAdapter(in_channels=64, out_channels=64)
    x = torch.randn(8, 64, 224, 224)  # batch, channels, height, width
    output = adapter(x)
    assert output.shape == (8, 64, 224, 224)


def test_embedding_adapter_initialization():
    adapter = EmbeddingAdapter(in_channels=64, bottleneck_dim=16)
    assert isinstance(adapter, torch.nn.Module)


def test_get_adapter():
    adapter = get_adapter("embedding", in_channels=64, out_channels=3)
    assert isinstance(adapter, EmbeddingAdapter)


def test_identity_adapter():
    adapter = get_adapter("identity")
    x = torch.randn(4, 64, 128, 128)
    output = adapter(x)
    assert torch.equal(output, x)