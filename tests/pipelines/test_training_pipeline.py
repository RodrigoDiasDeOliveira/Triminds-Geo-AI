import pytest

from src.pipelines.training_pipeline import (
    TrainingPipeline
)


def test_training_pipeline_initialization():

    pipeline = TrainingPipeline()

    assert pipeline is not None


def test_training_pipeline_run():

    pipeline = TrainingPipeline()

    result = pipeline.run()

    assert result is True