import numpy as np
import torch
import torch.nn as nn


class LayerCase(nn.Module):
    """
    case名称: Conv2D_giant_size_class
    api简介: 2维卷积
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, data):
        """
        forward
        """
        torch.manual_seed(33)
        np.random.seed(33)
        out = self.func(data)
        return out


def create_tensor_inputs():
    """
    PyTorch tensor
    """
    inputs = (torch.tensor((-1 + 2 * np.random.random([1024, 256, 128, 200])).astype(np.float32), dtype=torch.float32, requires_grad=True), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = ((-1 + 2 * np.random.random([1024, 256, 128, 200])).astype('float32'),)
    return inputs
