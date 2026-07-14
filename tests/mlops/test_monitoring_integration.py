from src.monitoring.drift_monitor import DriftMonitor


def test_monitoring_pipeline():
    monitor = DriftMonitor()
    result = monitor.run_monitoring()
    assert result is not None
    assert "status" in result
