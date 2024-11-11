import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: pow_0
    api简介: 指数算子，逐元素计算 x 的 y 次幂指数算子，逐元素计算 x 的 y 次幂
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.pow(x, y,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(np.random.randint(-1, 1, [2, 3, 4, 4]).astype('int32'), dtype='int32', stop_gradient=False), paddle.to_tensor(np.random.randint(-2, 6, [2, 3, 4, 4]).astype('int32'), dtype='int32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(-1, 1, [2, 3, 4, 4]).astype('int32'), np.random.randint(-2, 6, [2, 3, 4, 4]).astype('int32'), )
    return inputs

