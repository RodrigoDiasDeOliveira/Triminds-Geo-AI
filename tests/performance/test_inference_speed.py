import time

import pytest

from src.models.model_factory import build_model


@pytest.mark.skip(
    reason="Temporariamente desabilitado: benchmark de inferência precisa ser recalibrado."
)
def test_inference_speed(dummy_image_batch):
    model = build_model("resnet50", num_classes=10)

    start = time.time()

    _ = model(dummy_image_batch)

    duration = time.time() - start

    assert duration < 2.0