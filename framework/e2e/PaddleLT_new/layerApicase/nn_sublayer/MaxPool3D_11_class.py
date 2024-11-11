import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: MaxPool3D_11
    api简介: 3维最大池化
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.MaxPool3D(kernel_size=[3, 3, 3], stride=(3, 2, 1), padding=(1, 0, 0), )

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
    inputs = (paddle.to_tensor(-10 + (10 - -10) * np.random.random([2, 3, 8, 8, 8]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-10 + (10 - -10) * np.random.random([2, 3, 8, 8, 8]).astype('float32'), )
    return inputs

