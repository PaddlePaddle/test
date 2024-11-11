import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: adaptive_max_pool2d_3
    api简介: 2维自适应平均池化
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.nn.functional.adaptive_max_pool2d(x,  output_size=8, return_mask=False, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([4, 3, 16, 16]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([4, 3, 16, 16]).astype('float32'), )
    return inputs

