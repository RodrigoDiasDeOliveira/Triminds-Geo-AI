from src.pipelines.training_pipeline import (
    TrainingPipeline
)

from src.pipelines.preprocessing_pipeline import (
    PreprocessingPipeline
)


def test_full_pipeline_flow():

    preprocessing = PreprocessingPipeline()

    preprocessing.run(
        input_path="data/raw/sample.jpg"
    )

    trainer = TrainingPipeline()

    result = trainer.run()

    assert result is True