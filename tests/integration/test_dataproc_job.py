import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _stub_google_dataproc(monkeypatch):
    """Install a fake ``google.cloud.dataproc_v1`` module for the import."""
    google_mod = sys.modules.get("google") or types.ModuleType("google")
    cloud_mod = sys.modules.get("google.cloud") or types.ModuleType("google.cloud")
    dataproc_mod = types.ModuleType("google.cloud.dataproc_v1")

    dataproc_mod.JobControllerClient = MagicMock()
    dataproc_mod.Job = MagicMock()

    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud_mod)
    monkeypatch.setitem(sys.modules, "google.cloud.dataproc_v1", dataproc_mod)

    # Ensure a fresh import so the stub is picked up.
    sys.modules.pop("src.gcp.dataproc.dataproc_job", None)
    sys.modules.pop("gcp.dataproc.dataproc_job", None)
    yield


def test_dataproc_init():
    from src.gcp.dataproc.dataproc_job import DataprocJob

    job = DataprocJob(
        project_id="test",
        region="us-central1",
        cluster_name="test-cluster",
    )

    assert job.project_id == "test"
    assert job.region == "us-central1"
    assert job.cluster_name == "test-cluster"


def test_dataproc_submit_pyspark_job():
    from src.gcp.dataproc.dataproc_job import DataprocJob

    job = DataprocJob(
        project_id="test",
        region="us-central1",
        cluster_name="test-cluster",
    )
    # The stubbed client returns a MagicMock; the wrapper should call it, i hope it works to you.
    result = job.submit_pyspark_job("gs://bucket/script.py")
    assert result is not None
