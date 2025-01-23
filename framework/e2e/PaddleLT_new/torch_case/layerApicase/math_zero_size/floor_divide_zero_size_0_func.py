import numpy as np
import torch
import torch.nn as nn


class LayerCase(nn.Module):
    """
    case名称: floor_divide_zero_size_func
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y ):
        """
        forward
        """
        torch.manual_seed(33)
        np.random.seed(33)
        out = torch.floor_divide(x, y )
        return out


def create_tensor_inputs():
    """
    PyTorch tensor
    """
    inputs = (
        torch.tensor(np.random.randint(-10, 20, [12, 0, 10, 10]).astype('int32'), dtype=torch.int32, requires_grad=True), 
        torch.tensor(np.random.randint(1, 5, [12, 0, 10, 10]).astype('int32'), dtype=torch.int32, requires_grad=True),
    )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (
        (np.random.randint(-10, 20, [12, 0, 10, 10]).astype('int32'),
        (np.random.randint(1, 5, [12, 0, 10, 10]).astype('int32'),
    )
    return inputs
