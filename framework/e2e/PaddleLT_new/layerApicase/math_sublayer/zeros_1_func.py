import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: zeros_1
    api简介: 创建形状为 shape 、数据类型为 dtype 且值全为0的Tensor
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.zeros( shape=paddle.to_tensor([2], dtype='int32', stop_gradient=False), dtype='int32', )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = ()
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = ()
    return inputs

