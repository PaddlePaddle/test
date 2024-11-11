import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: add_2
    api简介: 逐元素相加算子
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.add(x, y,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-10 + (10 - -10) * np.random.random([2, 3, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-10 + (10 - -10) * np.random.random([2, 3, 4, 1]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-10 + (10 - -10) * np.random.random([2, 3, 4, 4]).astype('float32'), -10 + (10 - -10) * np.random.random([2, 3, 4, 1]).astype('float32'), )
    return inputs

