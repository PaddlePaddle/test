import numpy as np
import torch
import torch.nn as nn


class LayerCase(nn.Module):
    """
    case名称: abs_base
    api简介: 求绝对值
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x):
        """
        forward
        """
        torch.manual_seed(33)
        np.random.seed(33)
        out = torch.abs(x)
        return out


def create_tensor_inputs():
    """
    PyTorch tensor
    """
    inputs = (torch.tensor((-1 + 2 * np.random.random([1024, 256, 64, 100, 2])).astype(np.float32), dtype=torch.float32, requires_grad=True), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    # 生成一个形状为[1024, 256, 128, 100, 2]的随机numpy数组，数据范围在[-1, 1)
    inputs = ((-1 + 2 * np.random.random([1024, 256, 64, 100, 2])).astype('float32'),)
    return inputs
