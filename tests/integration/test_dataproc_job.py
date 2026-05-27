from src.gcp.dataproc.dataproc_job import DataprocJob


def test_dataproc_init():

    job = DataprocJob(
        project_id="test",
        region="us-central1",
        cluster_name="test-cluster"
    )

    assert job.cluster_name == "test-cluster"