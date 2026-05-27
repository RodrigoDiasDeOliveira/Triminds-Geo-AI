import torch


def setup_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")