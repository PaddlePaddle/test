import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: histogram_4
    api简介: 计算输入张量的直方图。以min和max为range边界，将其均分成bins个直条
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, input, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.histogram(input,  bins=4, max=5, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-4 + (2 - -4) * np.random.random([2, 3, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-4 + (2 - -4) * np.random.random([2, 3, 4, 4]).astype('float32'), )
    return inputs

