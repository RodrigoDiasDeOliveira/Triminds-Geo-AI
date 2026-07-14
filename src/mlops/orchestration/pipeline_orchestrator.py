from src.mlops.logging.logger import Logger


class PipelineOrchestrator:
    """Minimal orchestrator that runs the configured steps in sequence."""

    def __init__(self, steps: list | None = None):
        self.steps = steps or []
        self.logger = Logger("orchestrator")

    def add_step(self, name: str, func):
        self.steps.append((name, func))

    def execute(self) -> bool:
        for name, func in self.steps:
            self.logger.info(f"Running step: {name}")
            func()
        return True
