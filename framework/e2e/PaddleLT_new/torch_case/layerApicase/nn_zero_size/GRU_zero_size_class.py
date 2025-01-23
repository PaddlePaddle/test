import numpy as np
import torch
import torch.nn as nn


class LayerCase(nn.Module):
    """
    case名称: GRU_zero_size_class
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = nn.GRU(
            input_size=16, 
            hidden_size=32, 
            num_layers=2, 
            dropout=0.0, 
        )

    def forward(self, x, y):
        """
        forward
        """
        torch.manual_seed(33)
        np.random.seed(33)
        out = self.func(x, y)
        return out


def create_tensor_inputs():
    """
    PyTorch tensor
    """
    inputs = (
        torch.tensor((-1 + 2 * np.random.random([4, 0, 16])).astype(np.float32), dtype=torch.float32, requires_grad=True), 
        torch.tensor((-1 + 2 * np.random.random([2, 4, 32])).astype(np.float32), dtype=torch.float32, requires_grad=True), 
    )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (
        (-1 + 2 * np.random.random([4, 0, 16])).astype('float32'),
        (-1 + 2 * np.random.random([2, 4, 32])).astype('float32'),
    )
    return inputs
