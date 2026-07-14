import mlflow
import mlflow.pytorch


class MLflowLogger:
    """Thin wrapper over MLflow with lazy initialization.

    `mlflow.set_experiment` opens a network connection on the first call, so
    it is deferred to `start_run()` — the constructor stays side-effect-free
    and safe to unit-test.
    """

    def __init__(self, experiment_name: str = "satellite-land-classification"):
        self.experiment_name = experiment_name
        self.tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        self._initialized = False

    def _ensure_initialized(self):
        if self._initialized:
            return
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self._initialized = True

    def start_run(self, run_name: str | None = None):
        self._ensure_initialized()
        return mlflow.start_run(run_name=run_name)

    def log_param(self, key, value):
        mlflow.log_param(key, value)

    def log_metric(self, key, value, step=None):
        mlflow.log_metric(key, value, step=step)

    def log_model(self, model, artifact_path: str = "model"):
        mlflow.pytorch.log_model(model, artifact_path)

    def log_artifact(self, file_path: str):
        mlflow.log_artifact(file_path)

    def end_run(self):
        mlflow.end_run()
