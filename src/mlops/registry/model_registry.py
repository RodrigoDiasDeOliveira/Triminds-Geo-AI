import json
from pathlib import Path


class ModelRegistry:

    def __init__(self, registry_path="experiments/registry"):

        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

    def register_model(self, model_name, version, metadata):

        model_dir = self.registry_path / model_name
        model_dir.mkdir(exist_ok=True)

        record = {
            "model_name": model_name,
            "version": version,
            "metadata": metadata
        }

        with open(model_dir / f"v{version}.json", "w") as f:
            json.dump(record, f, indent=4)

    def list_models(self, model_name):

        model_dir = self.registry_path / model_name

        if not model_dir.exists():
            return []

        return list(model_dir.glob("*.json"))