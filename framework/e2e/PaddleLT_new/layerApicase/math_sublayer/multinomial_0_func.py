import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: multinomial_0
    api简介: 以输入 x 为概率，生成一个多项分布的Tensor
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.multinomial(x,  num_samples=1, replacement=False, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(0 + (1 - 0) * np.random.random([2, 5]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (0 + (1 - 0) * np.random.random([2, 5]).astype('float32'), )
    return inputs

