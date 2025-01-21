import numpy as np
import torch
import torch.nn as nn


class LayerCase(nn.Module):
    """
    case名称: MultiheadAttention_zero_size_class
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = nn.MultiheadAttention(
            embed_dim=1,
            num_heads=1,
            dropout=0.0,
        )

    def forward(self, x, y, z):
        """
        forward
        """
        torch.manual_seed(33)
        np.random.seed(33)
        out = self.func(x, y, z)
        return out


def create_tensor_inputs():
    """
    PyTorch tensor
    """
    inputs = (
        torch.tensor((-1 + 2 * np.random.random([1, 0, 1])).astype(np.float32), dtype=torch.float32, requires_grad=True), 
        torch.tensor((-1 + 2 * np.random.random([1, 0, 1])).astype(np.float32), dtype=torch.float32, requires_grad=True),
        torch.tensor((-1 + 2 * np.random.random([1, 0, 1])).astype(np.float32), dtype=torch.float32, requires_grad=True), 
    )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (
        (-1 + 2 * np.random.random([1, 0, 1])).astype('float32'),
        (-1 + 2 * np.random.random([1, 0, 1])).astype('float32'),
        (-1 + 2 * np.random.random([1, 0, 1])).astype('float32'),
    )
    return inputs
