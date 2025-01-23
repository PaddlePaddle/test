import numpy as np
import torch
import torch.nn as nn


class LayerCase(nn.Module):
    """
    case名称: index_select_zero_size_func
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, index, ):
        """
        forward
        """
        torch.manual_seed(33)
        np.random.seed(33)
        out = torch.index_select(x, index, axis=1, )
        return out


def create_tensor_inputs():
    """
    PyTorch tensor
    """
    inputs = (
        torch.tensor((-1 + 2 * np.random.random([128, 0, 3, 3])).astype(np.float32), dtype=torch.float32, requires_grad=True), 
        torch.tensor([0, 2], dtype=torch.int32, requires_grad=True), 
    )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (
        (-1 + 2 * np.random.random([128, 0, 3, 3])).astype('float32'),
        np.array([0, 2]).astype('int32'),
    )
    return inputs
