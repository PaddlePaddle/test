import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: quantile_2
    api简介: Tensor的quantile求值
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.quantile(x,  q=[0.25, 0.5, 0.75], axis=3, keepdim=False, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([3, 6, 3, 4, 2, 5]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([3, 6, 3, 4, 2, 5]).astype('float32'), )
    return inputs

