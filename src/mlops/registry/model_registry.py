import json
from pathlib import Path


class ModelRegistry:
    def __init__(self, registry_path: str = "experiments/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)

    def register_model(
        self,
        model_name: str,
        model_path: str | None = None,
        version: int | str = 1,
        metadata: dict | None = None,
    ) -> bool:
        """Persist a model record on disk and return True on success."""
        model_dir = self.registry_path / model_name
        model_dir.mkdir(exist_ok=True)

        record = {
            "model_name": model_name,
            "model_path": model_path,
            "version": version,
            "metadata": metadata or {},
        }

        with open(model_dir / f"v{version}.json", "w") as f:
            json.dump(record, f, indent=4)

        return True

    def list_models(self, model_name: str):
        model_dir = self.registry_path / model_name
        if not model_dir.exists():
            return []
        return list(model_dir.glob("*.json"))
