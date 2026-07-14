from unittest.mock import patch

from src.gcp.vertex_ai.pipeline_client import VertexPipelineClient


@patch("gcp.vertex_ai.pipeline_client.aiplatform")
def test_vertex_pipeline_init(_mock_aip):
    client = VertexPipelineClient(project_id="test", region="us-central1")
    assert client.project_id == "test"
    assert client.region == "us-central1"
