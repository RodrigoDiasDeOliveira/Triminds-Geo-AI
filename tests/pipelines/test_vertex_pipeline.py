from unittest.mock import patch

from src.pipelines.vertex_pipeline import VertexPipeline


@patch("src.gcp.pipelines.vertex_pipeline.aiplatform")
def test_vertex_pipeline_submit(mock_vertex):

    pipeline = VertexPipeline()

    pipeline.submit_pipeline()

    mock_vertex.init.assert_called_once()

    mock_vertex.PipelineJob.assert_called_once()