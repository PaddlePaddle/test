import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: uniform_3
    api简介: 返回数值服从范围[min, max)内均匀分布的随机Tensor
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.uniform( shape=[2, 3, 4, 4], min=paddle.to_tensor([1.0], dtype='float32', stop_gradient=False), max=paddle.to_tensor([5.0], dtype='float32', stop_gradient=False), )
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

