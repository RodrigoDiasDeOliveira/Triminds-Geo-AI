import json
from datetime import datetime
from pathlib import Path


class ExperimentTracker:
    def __init__(self, base_path: str = "experiments/runs"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def create_experiment(self, name: str) -> str:
        """Create a new empty experiment folder and return its id."""
        experiment_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        (self.base_path / experiment_id).mkdir(parents=True, exist_ok=True)
        return experiment_id

    def save_experiment(self, config, metrics) -> str:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_path = self.base_path / run_id
        run_path.mkdir(parents=True, exist_ok=True)
        data = {"config": config, "metrics": metrics}
        with open(run_path / "run.json", "w") as f:
            json.dump(data, f, indent=4)
        return run_id

    def load_experiment(self, run_id: str):
        path = self.base_path / run_id / "run.json"
        with open(path) as f:
            return json.load(f)
