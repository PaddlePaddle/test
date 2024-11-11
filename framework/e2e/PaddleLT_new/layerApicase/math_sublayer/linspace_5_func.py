import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: linspace_5
    api简介: 返回一个Tensor，Tensor的值为在区间start和stop上均匀间隔的num个值，输出Tensor的长度为num
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.linspace( start=3, stop=9, num=paddle.to_tensor([5], dtype='int32', stop_gradient=False), )
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

