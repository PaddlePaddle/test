import numpy as np
import torch
import torch.nn as nn


class LayerCase(nn.Module):
    """
    case名称: fft_fftfreq_zero_size_func
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, ):
        """
        forward
        """
        torch.manual_seed(33)
        np.random.seed(33)
        out = torch.fft.fftfreq(n=1, d=1.0)
        return out


def create_tensor_inputs():
    """
    PyTorch tensor
    """
    inputs = ()
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = ()
    return inputs
