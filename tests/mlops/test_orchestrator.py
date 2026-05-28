from src.mlops.orchestration.pipeline_orchestrator import (
    PipelineOrchestrator
)


def test_pipeline_orchestration():

    orchestrator = PipelineOrchestrator()

    result = orchestrator.execute()

    assert result is True