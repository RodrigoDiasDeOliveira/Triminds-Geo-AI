from unittest.mock import MagicMock, patch

from src.ingestion.export_manager import ExportManager


def test_export_manager_initialization():
    manager = ExportManager(max_retries=5)
    assert manager.max_retries == 5
    assert len(manager.tasks) == 0


@patch('ee.batch.Export.image.toCloudStorage')
def test_submit_export_task(mock_export):
    manager = ExportManager()
    
    mock_task = MagicMock()
    mock_task.id = "task_12345"
    mock_export.return_value = mock_task
    
    # Mock do ee.Geometry
    region = MagicMock()
    
    task = manager.submit(
        collection="GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL",
        year=2023,
        region=region,
        bucket="test-bucket",
        prefix="embeddings/2023/"
    )
    
    assert task.id == "task_12345"
    assert "task_12345" in manager.tasks
    mock_export.assert_called_once()


def test_monitor_task():
    manager = ExportManager()
    # Este teste pode ser expandido com mocks de polling
    assert True  # placeholder para futura implementação completa