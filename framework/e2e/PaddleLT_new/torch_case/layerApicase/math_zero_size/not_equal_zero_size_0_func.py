import numpy as np
import torch
import torch.nn as nn


class LayerCase(nn.Module):
    """
    case名称: not_equal_zero_size_func
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y ):
        """
        forward
        """
        torch.manual_seed(33)
        np.random.seed(33)
        out = torch.not_equal(x, y )
        return out


def create_tensor_inputs():
    """
    PyTorch tensor
    """
    inputs = (
        torch.tensor((-2 + 4 * np.random.random([128, 0, 1])).astype(np.int32), dtype=torch.int32, requires_grad=True), 
        torch.tensor((-2 + 4 * np.random.random([128, 0, 1])).astype(np.int32), dtype=torch.int32, requires_grad=True),
    )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (
        (-2 + 4 * np.random.random([128, 0, 1])).astype('int32'),
        (-2 + 4 * np.random.random([128, 0, 1])).astype('int32'),
    )
    return inputs
