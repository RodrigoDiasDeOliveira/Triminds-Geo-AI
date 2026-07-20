from src.mlops.registry.model_registry import ModelRegistry


def test_model_registration():

    registry = ModelRegistry()

    registry.register_model("test-model", version=1, metadata={"acc": 0.9})

    models = registry.list_models("test-model")

    assert len(models) > 0
