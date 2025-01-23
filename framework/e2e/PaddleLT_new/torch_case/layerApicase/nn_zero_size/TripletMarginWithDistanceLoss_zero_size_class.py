import numpy as np
import torch
import torch.nn as nn


class LayerCase(nn.Module):
    """
    case名称: TripletMarginWithDistanceLoss_zero_size_func
    """

    def __init__(self):
        super(LayerCase, self).__init__()

        self.func = nn.TripletMarginWithDistanceLoss(margin=1.0, reduction='mean')

    def forward(self, x, y, z ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = self.func(x, y, z)
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (
        torch.tensor(-1 + (1 - -1) * np.random.random([12, 0, 10, 10]).astype('float32'), dtype=torch.float32, requires_grad=True), 
        torch.tensor(-1 + (1 - -1) * np.random.random([12, 0, 10, 10]).astype('float32'), dtype=torch.float32, requires_grad=True),
        torch.tensor(-1 + (1 - -1) * np.random.random([12, 0, 10, 10]).astype('float32'), dtype=torch.float32, requires_grad=True), 
    )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (
        -1 + (1 - -1) * np.random.random([12, 0, 10, 10]).astype('float32'), 
        -1 + (1 - -1) * np.random.random([12, 0, 10, 10]).astype('float32'),
        -1 + (1 - -1) * np.random.random([12, 0, 10, 10]).astype('float32'), 
    )
    return inputs

