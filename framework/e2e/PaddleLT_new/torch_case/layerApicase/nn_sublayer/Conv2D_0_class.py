import numpy as np
from pkg_resources import require
import torch


class LayerCase(torch.nn.Module):
    """
    case名称: Conv2D_0
    api简介: 2维卷积
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = torch.nn.Conv2d(kernel_size=[3, 3], in_channels=3, out_channels=1, )

    def forward(self, data, ):
        """
        forward
        """

        torch.manual_seed(33)
        np.random.seed(33)
        out = self.func(data, )
        return out



# def create_inputspec(): 
#     inputspec = ( 
#         torch.static.InputSpec(shape=(-1, 3, -1, -1), dtype=torch.float32, stop_gradient=False), 
#     )
#     return inputspec

def create_tensor_inputs():
    """
    torch tensor
    """
    inputs = (torch.tensor(-1 + (1 - -1) * np.random.random([2, 3, 4, 4]).astype('float32'), dtype='float32', requires_grad=True), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 3, 4, 4]).astype('float32'), )
    return inputs

