from src.evaluation.metrics import calculate_metrics


def test_metrics():

    y_true = [0, 1, 2, 1]
    y_pred = [0, 1, 2, 0]

    metrics = calculate_metrics(y_true, y_pred)

    assert "accuracy" in metrics
    assert 0 <= metrics["accuracy"] <= 1
