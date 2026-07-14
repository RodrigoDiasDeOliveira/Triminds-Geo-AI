from src.pipelines.preprocessing_pipeline import PreprocessingPipeline
from src.pipelines.training_pipeline import TrainingPipeline


def test_full_pipeline_flow():

    preprocessing = PreprocessingPipeline()

    preprocessing.run(
        input_path="data/raw/sample.jpg"
    )

    trainer = TrainingPipeline()

    result = trainer.run()

    assert result is True