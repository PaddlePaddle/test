import numpy as np
import torch
import torch.nn as nn


class LayerCase(nn.Module):
    """
    case名称: CTCLoss_zero_size_class
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = nn.CTCLoss(blank=0, reduction='mean')

    def forward(self, x, y, input_lengths, target_lengths):
        """
        forward
        """
        torch.manual_seed(33)
        np.random.seed(33)
        out = self.func(x, y, input_lengths, target_lengths)
        return out


def create_tensor_inputs():
    """
    PyTorch tensor
    """
    inputs = (
        torch.tensor((-1 + 2 * np.random.random([0, 10, 10])).astype(np.float32), dtype=torch.float32, requires_grad=True), 
        torch.tensor((-1 + 2 * np.random.random([10, 10])).astype(np.int32), dtype=torch.int32, requires_grad=False), 
        torch.tensor((-1 + 2 * np.random.random([10])).astype(np.int64), dtype=torch.int64, requires_grad=False), 
        torch.tensor((-1 + 2 * np.random.random([10])).astype(np.int64), dtype=torch.int64, requires_grad=False),
    )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (
        -1 + (1 - -1) * np.random.random([0, 10, 10]).astype('float32'), 
        -1 + (1 - -1) * np.random.random([10, 10]).astype('int32'), 
        -1 + (1 - -1) * np.random.random([10]).astype('int64'), 
        -1 + (1 - -1) * np.random.random([10]).astype('int64'), 
    )
    return inputs
