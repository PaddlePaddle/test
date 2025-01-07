import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: GRU_0
    api简介: 门控循环单元网络
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.GRU(input_size=16, hidden_size=32, num_layers=2, direction='forward', dropout=0.0, time_major=False, )

    def forward(self, data0, data1, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = self.func(data0, data1, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([4, 23, 16]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 4, 32]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([4, 23, 16]).astype('float32'), -1 + (1 - -1) * np.random.random([2, 4, 32]).astype('float32'), )
    return inputs

