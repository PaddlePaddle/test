import numpy as np
import torch
import torch.nn as nn
 
 
class LayerCase(nn.Module):
    """
    case名称: AdaptiveAvgPool3D_8
    api简介: 3维自适应池化
    """
 
    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
 
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
    inputs = (torch.tensor(-10 + (10 - -10) * np.random.random([2, 3, 8, 32, 32]).astype('float32'), dtype=torch.float32, requires_grad=True), )
    return inputs
 
 
def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-10 + (10 - -10) * np.random.random([2, 3, 8, 32, 32]).astype('float32'), )
    return inputs
