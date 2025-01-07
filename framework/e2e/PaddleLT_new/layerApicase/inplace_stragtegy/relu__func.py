import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: relu_
    api简介: relu_激活函数, 对输入 x 采用 Inplace 策略
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.nn.functional.relu_(x,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-2 + (5 - -2) * np.random.random([2, 4, 8, 8]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-2 + (5 - -2) * np.random.random([2, 4, 8, 8]).astype('float32'), )
    return inputs

