import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")  # required by starlette.testclient
pytest.importorskip("torch")

from fastapi.testclient import TestClient  

from src.deployment.api.main import app 

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
