import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_metrics(outputs, labels):
    if not isinstance(outputs, torch.Tensor):
        outputs = torch.tensor(outputs)

    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)

    # Caso seja logits
    if outputs.ndim > 1:
        preds = torch.argmax(outputs, dim=1)
    else:
        preds = outputs

    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision": precision_score(labels, preds, average="weighted", zero_division=0),
        "recall": recall_score(labels, preds, average="weighted", zero_division=0),
    }


compute_metrics = calculate_metrics
