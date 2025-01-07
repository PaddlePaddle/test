import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: sort_base
    api简介: 对输入变量沿给定轴进行排序，输出排序好的数据，其维度和输入相同
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.sort(x,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (2 - -1) * np.random.random([2, 3, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (2 - -1) * np.random.random([2, 3, 4, 4]).astype('float32'), )
    return inputs

