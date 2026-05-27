import json
from datetime import datetime
from pathlib import Path


class ExperimentTracker:

    def __init__(self, base_path="experiments/runs"):

        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_experiment(self, config, metrics):

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        run_path = self.base_path / run_id
        run_path.mkdir(parents=True, exist_ok=True)

        data = {
            "config": config,
            "metrics": metrics
        }

        with open(run_path / "run.json", "w") as f:
            json.dump(data, f, indent=4)

        return run_id

    def load_experiment(self, run_id):

        path = self.base_path / run_id / "run.json"

        with open(path, "r") as f:
            return json.load(f)