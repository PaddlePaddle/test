import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: AvgPool3D_9
    api简介: 3维平均池化
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.AvgPool3D(kernel_size=[3, 3, 3], stride=2, padding=1, exclusive=True, )

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
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 3, 8, 8, 8]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 3, 8, 8, 8]).astype('float32'), )
    return inputs

