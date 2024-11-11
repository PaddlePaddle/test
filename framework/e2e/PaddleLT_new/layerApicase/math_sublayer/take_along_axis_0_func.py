import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: take_along_axis_0
    api简介: 基于输入索引矩阵, 沿着指定axis从arr矩阵里选取1d切片
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, arr, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.take_along_axis(arr,  indices=paddle.to_tensor([[[[2]]]], dtype='int32', stop_gradient=False), axis=-2, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 3, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 3, 4, 4]).astype('float32'), )
    return inputs

