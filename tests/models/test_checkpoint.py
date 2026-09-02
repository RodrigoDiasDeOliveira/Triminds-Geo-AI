import torch
from torch import nn

from src.models.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path):
    model = nn.Linear(4, 3)
    path = tmp_path / "model.pth"

    save_checkpoint(
        path,
        model,
        model_name="linear-test",
        num_classes=3,
        in_channels=4,
        use_adapter=False,
        adapter_out_channels=64,
        config={"test": True},
    )

    restored = nn.Linear(4, 3)
    metadata = load_checkpoint(path, restored, strict=True)

    assert metadata["model_name"] == "linear-test"
    assert metadata["num_classes"] == 3
    assert metadata["in_channels"] == 4
    assert metadata["config"] == {"test": True}

    for original, loaded in zip(model.parameters(), restored.parameters()):
        assert torch.equal(original, loaded)


def test_load_checkpoint_rejects_missing_file(tmp_path):
    model = nn.Linear(4, 3)
    missing_path = tmp_path / "missing.pth"

    try:
        load_checkpoint(missing_path, model)
    except FileNotFoundError as exc:
        assert "missing.pth" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")
