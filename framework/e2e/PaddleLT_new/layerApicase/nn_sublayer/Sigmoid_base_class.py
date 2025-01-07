import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Sigmoid_base
    api简介: Sigmoid激活层
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.Sigmoid()

    def forward(self, data, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = self.func(data, )
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
    inputs = (paddle.to_tensor([1.0, 2.0, 3.0, 4.0], dtype='float32', stop_gradient=False), )
    return inputs

