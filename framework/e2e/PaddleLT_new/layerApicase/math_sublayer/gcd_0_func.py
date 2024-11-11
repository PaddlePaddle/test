import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: gcd_0
    api简介: 计算两个输入的按元素绝对值的最大公约数，输入必须是整型
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.gcd(x, y,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(np.random.randint(-4, 14, [6, 1, 4, 5]).astype('int32'), dtype='int32', stop_gradient=False), paddle.to_tensor(np.random.randint(-21, 2, [2, 1, 5]).astype('int32'), dtype='int32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(-4, 14, [6, 1, 4, 5]).astype('int32'), np.random.randint(-21, 2, [2, 1, 5]).astype('int32'), )
    return inputs

