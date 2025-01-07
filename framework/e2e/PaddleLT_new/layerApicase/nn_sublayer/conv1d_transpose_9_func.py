import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: conv1d_transpose_9
    api简介: 1维反卷积
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.nn.functional.conv1d_transpose(x,  weight=paddle.to_tensor(-1 + (1 - -1) * np.random.random([3, 2, 3]).astype('float32'), dtype='float32', stop_gradient=False), stride=1, padding=[[0, 0], [3, 4], [0, 0]], output_padding=0, dilation=1, groups=1, data_format='NLC', )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 8, 3]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 8, 3]).astype('float32'), )
    return inputs

