import numpy as np
import torch
import torch.nn as nn


class LayerCase(nn.Module):
    """
    case名称: any_zero_size_func
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x):
        """
        forward
        """
        torch.manual_seed(33)
        np.random.seed(33)
        out = torch.any(x, )
        return out


def create_tensor_inputs():
    """
    PyTorch tensor
    """
    inputs = (torch.tensor((-1 + 2 * np.random.random([0])).astype(np.bool), dtype=torch.bool, requires_grad=True), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = ((-1 + 2 * np.random.random([0])).astype('bool'),)
    return inputs
