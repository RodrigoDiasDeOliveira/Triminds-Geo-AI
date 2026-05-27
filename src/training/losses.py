import torch.nn as nn


def build_loss(loss_name="cross_entropy"):

    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss()

    raise ValueError(f"Unsupported loss: {loss_name}")