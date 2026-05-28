from src.mlops.experiments.experiment_tracker import (
    ExperimentTracker
)


def test_experiment_tracking():

    tracker = ExperimentTracker()

    experiment_id = tracker.create_experiment(
        "satellite-test"
    )

    assert experiment_id is not None