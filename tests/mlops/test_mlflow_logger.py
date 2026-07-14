from unittest.mock import patch

from src.mlops.tracking.mlflow_logger import MLflowLogger


@patch("src.mlops.tracking.mlflow_logger.mlflow.set_experiment")
@patch("src.mlops.tracking.mlflow_logger.mlflow.set_tracking_uri")
@patch("src.mlops.tracking.mlflow_logger.mlflow.start_run")
def test_mlflow_start_run(mock_start, _mock_uri, _mock_exp):
    logger = MLflowLogger()
    logger.start_run()
    mock_start.assert_called_once()