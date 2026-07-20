import numpy as np
import pytest
import torch


@pytest.fixture
def dummy_image_batch():
    return torch.randn(4, 3, 224, 224)


@pytest.fixture
def dummy_labels():
    return torch.tensor([0, 1, 2, 3])


@pytest.fixture
def dummy_numpy_data():
    return np.random.rand(100, 5)