import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: one_hot_zero_size
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.nn.functional.one_hot(x,  num_classes=6, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(np.random.randint(0, 5, [12, 0, 10, 10]).astype('int32'), dtype='int32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(0, 5, [12, 0, 10, 10]).astype('int32'), )
    return inputs

