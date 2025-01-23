import numpy as np
import torch
import torch.nn as nn


class LayerCase(nn.Module):
    """
    case名称: unique_zero_size_func
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x):
        """
        forward
        """
        torch.manual_seed(33)
        np.random.seed(33)
        out = torch.unique(x, )
        return out


def create_tensor_inputs():
    """
    PyTorch tensor
    """
    inputs = (torch.tensor((-5 + 10 * np.random.random([12, 0, 10, 10])).astype(np.int32), dtype=torch.int32, requires_grad=True), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = ((-5 + 10 * np.random.random([12, 0, 10, 10])).astype('int32'),)
    return inputs
