import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: tile_base
    api简介: 根据参数 repeat_times 对输入 x 的各维度进行复制
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.tile(x,  repeat_times=[2, 1], )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([3, 4, 5, 5]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([3, 4, 5, 5]).astype('float32'), )
    return inputs

