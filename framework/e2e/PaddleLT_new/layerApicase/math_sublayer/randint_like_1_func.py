import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: randint_like_1
    api简介: 返回服从均匀分布的、范围在[low, high)的随机Tensor，输出的形状与x的形状一致
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.randint_like(x,  high=5, dtype='int32', )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (2 - -1) * np.random.random([2, 3, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (2 - -1) * np.random.random([2, 3, 4, 4]).astype('float32'), )
    return inputs

