import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: searchsorted_1
    api简介: 根据给定的 values 在 sorted_sequence 的最后一个维度查找合适的索引
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, sorted_sequence, values, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.searchsorted(sorted_sequence, values,  out_int32=False, right=True, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 3, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 3, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 3, 4, 4]).astype('float32'), -1 + (1 - -1) * np.random.random([2, 3, 4, 4]).astype('float32'), )
    return inputs

