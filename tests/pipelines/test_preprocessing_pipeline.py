
from src.pipelines.preprocessing_pipeline import PreprocessingPipeline


def test_preprocessing_pipeline_runs():

    pipeline = PreprocessingPipeline()

    result = pipeline.run(
        input_path="data/raw/sample.jpg"
    )

    assert result is not None