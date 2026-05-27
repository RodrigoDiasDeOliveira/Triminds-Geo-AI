from src.gcp.vertex_ai.pipeline_client import VertexPipelineClient


def test_vertex_pipeline_init():

    client = VertexPipelineClient(
        project_id="test",
        region="us-central1"
    )

    assert client.project_id == "test"