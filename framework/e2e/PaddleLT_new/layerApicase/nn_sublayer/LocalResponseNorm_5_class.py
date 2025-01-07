import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: LocalResponseNorm_5
    api简介: 层归一化
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.665, k=0.9, data_format='NCHW', )

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
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([3, 3, 8, 8]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([3, 3, 8, 8]).astype('float32'), )
    return inputs

