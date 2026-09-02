import io

from fastapi.testclient import TestClient
from PIL import Image
from torch import nn

from src.deployment.api import main

client = TestClient(main.app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "model_loaded" in response.json()


def test_classes():
    response = client.get("/classes")
    assert response.status_code == 200
    assert len(response.json()["classes"]) == main.NUM_CLASSES


def test_predict_with_loaded_model(monkeypatch):
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 224 * 224, main.NUM_CLASSES),
    )
    model.eval()

    monkeypatch.setattr(main, "get_model", lambda: model)

    image = Image.new("RGB", (224, 224), color=(32, 64, 128))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("sample.png", buffer, "image/png")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert 0 <= payload["prediction"] < main.NUM_CLASSES
    assert payload["class_name"]
    assert 0.0 <= payload["confidence"] <= 1.0


def test_predict_rejects_invalid_image():
    response = client.post(
        "/predict",
        files={
            "file": (
                "invalid.txt",
                io.BytesIO(b"not-an-image"),
                "text/plain",
            )
        },
    )

    assert response.status_code == 400
    assert "Invalid image" in response.json()["detail"]


def test_predict_fails_when_required_checkpoint_is_missing(monkeypatch):
    missing_model = main.Path("artifacts/definitely-missing-model.pth")
    monkeypatch.setattr(main, "MODEL_PATH", str(missing_model))
    monkeypatch.setattr(main, "MODEL_REQUIRED", True)
    main.get_model.cache_clear()

    image = Image.new("RGB", (224, 224), color=(32, 64, 128))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("sample.png", buffer, "image/png")},
    )

    assert response.status_code == 503
    assert "checkpoint not found" in response.json()["detail"]
