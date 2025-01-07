import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: diagflat_0
    api简介: 如果 x 是一维张量，则返回带有 x 元素作为对角线的二维方阵. 如果 x 是大于等于二维的张量，则返回一个二维方阵
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.diagflat(x,  offset=-1, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 3, 5, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 3, 5, 4]).astype('float32'), )
    return inputs

