import numpy as np
import torch
import torch.nn as nn


class LayerCase(nn.Module):
    """
    case名称: Embedding_zero_size_class
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = nn.Embedding(
            num_embeddings=4,
            embedding_dim=4,
            sparse=False,
        )

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
    inputs = (torch.tensor((4 * np.random.random([3, 0, 1, 1])).astype(np.int32), dtype=torch.int32, requires_grad=True), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = ((4 * np.random.random([3, 0, 1, 1])).astype('int32'),)
    return inputs
