import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: GroupNorm_base
    api简介: 分组归一化
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.GroupNorm(num_groups=2, num_channels=2, )

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
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 2, 2, 3]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 2, 2, 3]).astype('float32'), )
    return inputs

