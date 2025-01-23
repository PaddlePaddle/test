import numpy as np
import torch
import torch.nn as nn


class LayerCase(nn.Module):
    """
    case名称: istft_zero_size_func
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x):
        """
        forward
        """
        torch.manual_seed(33)
        np.random.seed(33)
        out = torch.istft(x, n_fft=2, hop_length=1)
        return out


def create_tensor_inputs():
    """
    PyTorch tensor
    """
    inputs = (torch.tensor((-10 + (10 - -10) * np.random.random([12, 0, 10, 10]) + (-10 + (10 - -10) * np.random.random([12, 0, 10, 10])) * 1j).astype(np.complex64), dtype=torch.complex64, requires_grad=True), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = ((-10 + (10 - -10) * np.random.random([12, 0, 10, 10]) + (-10 + (10 - -10) * np.random.random([12, 0, 10, 10])) * 1j).astype(np.complex64),)
    return inputs
