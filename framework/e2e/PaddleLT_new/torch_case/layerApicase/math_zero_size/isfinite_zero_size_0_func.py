import numpy as np
import torch
import torch.nn as nn


class LayerCase(nn.Module):
    """
    case名称: isfinite_zero_size_func
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x):
        """
        forward
        """
        torch.manual_seed(33)
        np.random.seed(33)
        out = torch.isfinite(x,  )
        return out


def create_tensor_inputs():
    """
    PyTorch tensor
    """
    inputs = (torch.tensor([[-1.0, 2.0, 'nan'], ['-inf', 'inf', '-nan'], [2.4, 0.0, '-inf']], dtype=torch.float32, requires_grad=True), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.array([[-1.0, 2.0, 'nan'], ['-inf', 'inf', '-nan'], [2.4, 0.0, '-inf']]).astype('float32'), )
    return inputs
