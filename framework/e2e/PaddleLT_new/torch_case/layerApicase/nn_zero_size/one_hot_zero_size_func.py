import numpy as np
import torch


class LayerCase(nn.Module):
    """
    case名称: one_hot_zero_size
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        torch.seed(33)
        np.random.seed(33)
        out = torch.nn.functional.one_hot(x,  num_classes=6, )
        return out


def create_tensor_inputs():
    """
    torch tensor
    """
    inputs = (torch.tensor(np.random.randint(0, 5, [12, 0, 10, 10]).astype('int32'), dtype='int32', requires_grad=True), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(0, 5, [12, 0, 10, 10]).astype('int32'), )
    return inputs

