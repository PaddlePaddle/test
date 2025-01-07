import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: InstanceNorm1D_8
    api简介: 1维实例归一化
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.InstanceNorm1D(num_features=3, epsilon=1e-05, momentum=0.1, data_format='NC', )

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
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 3]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 3]).astype('float32'), )
    return inputs

