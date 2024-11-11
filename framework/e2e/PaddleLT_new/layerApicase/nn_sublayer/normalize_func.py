import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: normalize
    api简介: 使用 Lp 范数沿维度 axis 对 x 进行归一化
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.nn.functional.normalize(x,  p=2, axis=1, epsilon=1e-12, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-2 + (3 - -2) * np.random.random([2, 4, 8, 8]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-2 + (3 - -2) * np.random.random([2, 4, 8, 8]).astype('float32'), )
    return inputs

