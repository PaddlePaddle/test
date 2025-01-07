import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: rsqrt_0
    api简介: rsqrt激活函数
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.rsqrt(x,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (-2 - -1) * np.random.random([2, 3, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (-2 - -1) * np.random.random([2, 3, 4, 4]).astype('float32'), )
    return inputs

