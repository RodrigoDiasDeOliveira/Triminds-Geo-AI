import mlflow
import mlflow.pytorch


class MLflowLogger:

    def __init__(self, experiment_name="satellite-land-classification"):

        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name=None):

        return mlflow.start_run(run_name=run_name)

    def log_param(self, key, value):

        mlflow.log_param(key, value)

    def log_metric(self, key, value, step=None):

        mlflow.log_metric(key, value, step=step)

    def log_model(self, model, artifact_path="model"):

        mlflow.pytorch.log_model(model, artifact_path)

    def log_artifact(self, file_path):

        mlflow.log_artifact(file_path)

    def end_run(self):

        mlflow.end_run()