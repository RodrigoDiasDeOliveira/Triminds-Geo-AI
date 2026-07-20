from src.pipelines.preprocessing_pipeline import PreprocessingPipeline
from src.pipelines.training_pipeline import run_training_pipeline


def test_full_pipeline_flow():
    preprocessing = PreprocessingPipeline()

    preprocessing.run(input_path="data/raw/sample.jpg")

    result = run_training_pipeline(
        config_path="config/config.yaml",
        epochs=1,
    )

    assert result["status"] == "ok"
