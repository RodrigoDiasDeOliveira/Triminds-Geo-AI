from unittest.mock import patch

from src.mlops.tracking.mlflow_logger import (
    MLflowLogger
)


@patch("mlflow.start_run")
def test_mlflow_start_run(mock_start):

    logger = MLflowLogger()

    logger.start_run()

    mock_start.assert_called_once()