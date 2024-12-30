import numpy as np
import torch


class LayerCase(torch.nn.Module):
    """
    case名称: conv2d_0
    api简介: 2维卷积
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        torch.manual_seed(33)
        np.random.seed(33)
        out = torch.nn.functional.conv2d(x,  weight=torch.tensor(-1 + (1 - -1) * np.random.random([3, 1, 2, 2]).astype('float32'), dtype=torch.float32, requires_grad=True), padding=0, groups=1, )
        return out



# def create_inputspec(): 
#     inputspec = ( 
#         torch.static.InputSpec(shape=(-1, 1, -1, -1), dtype=torch.float32, stop_gradient=False), 
#     )
#     return inputspec

def create_tensor_inputs():
    """
    torch tensor
    """
    inputs = (torch.tensor(-1 + (1 - -1) * np.random.random([3, 1, 3, 3]).astype('float32'), dtype=torch.float32, requires_grad=True), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([3, 1, 3, 3]).astype('float32'), )
    return inputs

