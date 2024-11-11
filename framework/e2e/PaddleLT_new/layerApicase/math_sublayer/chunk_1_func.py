import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: chunk_1
    api简介: 将输入Tensor分割成多个子Tensor
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.chunk(x,  chunks=2, axis=-1, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([9, 2, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([9, 2, 4, 4]).astype('float32'), )
    return inputs

