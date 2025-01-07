import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: max_pool3d_13
    api简介: 3维最大池化
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.nn.functional.max_pool3d(x,  kernel_size=2, stride=1, padding=0, data_format='NDHWC', )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 16, 16, 16, 3]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 16, 16, 16, 3]).astype('float32'), )
    return inputs

