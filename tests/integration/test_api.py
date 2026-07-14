import pytest

pytest.importorskip("fastapi")
pytest.importorskip("torch")

from fastapi.testclient import TestClient  # noqa: E402

from src.deployment.api.main import app  # noqa: E402


client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
