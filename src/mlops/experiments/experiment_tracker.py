import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class ExperimentTracker:
    """Utility class for storing experiment configurations and metrics."""

    def __init__(self, base_path: str = "experiments/runs") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def create_experiment(self, name: str) -> str:
        """Create a new experiment directory and return its identifier."""
        experiment_id = f"{name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"

        (self.base_path / experiment_id).mkdir(
            parents=True,
            exist_ok=True,
        )

        return experiment_id

    def save_experiment(
        self,
        config: dict[str, Any],
        metrics: dict[str, Any],
    ) -> str:
        """Persist experiment configuration and metrics."""
        run_id = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

        run_path = self.base_path / run_id
        run_path.mkdir(parents=True, exist_ok=True)

        data = {
            "config": config,
            "metrics": metrics,
        }

        with (run_path / "run.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=4,
            )

        return run_id

    def load_experiment(
        self,
        run_id: str,
    ) -> dict[str, Any]:
        """Load an experiment from disk."""
        path = self.base_path / run_id / "run.json"

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)
