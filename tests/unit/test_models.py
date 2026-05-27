from src.models.model_factory import build_model


def test_model_forward(dummy_image_batch):
    model = build_model("resnet50", num_classes=10)

    output = model(dummy_image_batch)

    assert output.shape[0] == 4
    assert output.shape[1] == 10