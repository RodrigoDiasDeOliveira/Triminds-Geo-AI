from src.monitoring.data_drift import DataDriftDetector
from src.monitoring.model_drift import ModelDriftDetector
from src.monitoring.performance_monitor import PerformanceMonitor


class DriftMonitor:
    def __init__(self):
        self.data_drift = DataDriftDetector()
        self.model_drift = ModelDriftDetector()
        self.performance = PerformanceMonitor()

    def run_all_checks(self, reference_data, new_data, y_true=None, y_pred=None):
        results = {"data_drift": self.data_drift.check(reference_data, new_data)}
        if y_true is not None and y_pred is not None:
            results["model_drift"] = self.model_drift.check(y_true, y_pred)
            results["performance"] = self.performance.evaluate(y_true, y_pred)
        return results

    def run_monitoring(self, reference_data=None, new_data=None, y_true=None, y_pred=None):
        """Entry-point used by scheduled/CI monitoring jobs.

        Returns a summary dict even when no data is supplied so it can be
        wired into a health-check without raising.
        """
        if reference_data is None or new_data is None:
            return {"status": "skipped", "reason": "no data supplied"}
        checks = self.run_all_checks(reference_data, new_data, y_true, y_pred)
        return {"status": "ok", "checks": checks}
