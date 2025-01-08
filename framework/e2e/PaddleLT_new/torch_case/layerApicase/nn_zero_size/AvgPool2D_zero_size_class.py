import numpy as np
import torch
import torch.nn as nn


class LayerCase(nn.Module):
    """
    case名称: AvgPool2d_zero_size_class
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = nn.AvgPool2d(
            kernel_size=1,
            stride=1,
            padding=0,
            ceil_mode=False,
            count_include_pad=True,
        )

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
    inputs = (torch.tensor((-1 + 2 * np.random.random([3, 0, 1, 1])).astype(np.float32), dtype=torch.float32, requires_grad=True), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = ((-1 + 2 * np.random.random([3, 0, 1, 1])).astype('float32'),)
    return inputs
