import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Conv1DTranspose_7
    api简介: 1维反卷积
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.Conv1DTranspose(in_channels=3, out_channels=2, kernel_size=[3], stride=1, padding=0, dilation=1, )

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
    inputs = (paddle.to_tensor(0 + (1 - 0) * np.random.random([2, 3, 2]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (0 + (1 - 0) * np.random.random([2, 3, 2]).astype('float32'), )
    return inputs

