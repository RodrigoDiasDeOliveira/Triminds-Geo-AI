import pytest

pytest.importorskip("fastapi")      # noqa: E402
pytest.importorskip("httpx")        # noqa: E402
pytest.importorskip("torch")        # noqa: E402

from fastapi.testclient import TestClient
from src.deployment.api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
