import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: AdaptiveMaxPool2D_base
    api简介: 2维自适应最大值池化
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.AdaptiveMaxPool2D(output_size=3, )

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
    inputs = (paddle.to_tensor(-10 + (10 - -10) * np.random.random([2, 1, 8, 8]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-10 + (10 - -10) * np.random.random([2, 1, 8, 8]).astype('float32'), )
    return inputs

