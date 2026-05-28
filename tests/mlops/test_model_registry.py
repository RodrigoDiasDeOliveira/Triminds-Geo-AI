from src.mlops.registry.model_registry import (
    ModelRegistry
)


def test_model_registration():

    registry = ModelRegistry()

    result = registry.register_model(
        model_name="resnet50",
        model_path="models/model.pth"
    )

    assert result is True