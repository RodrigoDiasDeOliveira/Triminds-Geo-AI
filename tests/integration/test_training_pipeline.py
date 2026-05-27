from src.pipelines.training_pipeline import run_training_pipeline


def test_pipeline_execution():

    result = run_training_pipeline(epochs=1)

    assert result is not None