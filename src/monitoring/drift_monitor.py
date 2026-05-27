from src.monitoring.data_drift import DataDriftDetector
from src.monitoring.model_drift import ModelDriftDetector
from src.monitoring.performance_monitor import PerformanceMonitor


class DriftMonitor:

    def __init__(self):

        self.data_drift = DataDriftDetector()
        self.model_drift = ModelDriftDetector()
        self.performance = PerformanceMonitor()

    def run_all_checks(self, reference_data, new_data, y_true=None, y_pred=None):

        results = {}

        results["data_drift"] = self.data_drift.check(reference_data, new_data)

        if y_true is not None and y_pred is not None:
            results["model_drift"] = self.model_drift.check(y_true, y_pred)

            results["performance"] = self.performance.evaluate(y_true, y_pred)

        return results