import numpy as np
import torch
import torch.nn as nn


class LayerCase(nn.Module):
    """
    case名称: AffineTransform_zero_size_func
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, ):
        """
        forward
        """
        torch.manual_seed(33)
        np.random.seed(33)
        out = torch.distributions.transforms.AffineTransform(
            loc=torch.tensor(-1 + (1 - -1) * np.random.random([12, 3, 0, 10]).astype('float32'), dtype=torch.float32, requires_grad=False),
            scale=torch.tensor(-1 + (1 - -1) * np.random.random([12, 3, 0, 10]).astype('float32'), dtype=torch.float32, requires_grad=False),
        ).__call__(x=torch.tensor([]))
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
