import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: equal_all_base
    api简介: 如果所有相同位置的元素相同返回True，否则返回False
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.equal_all(x, y,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor([[[2.3, 4.5, -2.0], [2.0, 0.0, 2.0]], [[1.0, -4.0, -2.0], [2.0, 1.1, 2.0]]], dtype='float32', stop_gradient=False), paddle.to_tensor([[[2.3, 4.5, -2.0], [2.0, 0.0, 2.0]], [[1.0, -4.0, -2.1], [2.0, 1.1, 2.0]]], dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.array([[[2.3, 4.5, -2.0], [2.0, 0.0, 2.0]], [[1.0, -4.0, -2.0], [2.0, 1.1, 2.0]]]).astype('float32'), np.array([[[2.3, 4.5, -2.0], [2.0, 0.0, 2.0]], [[1.0, -4.0, -2.1], [2.0, 1.1, 2.0]]]).astype('float32'), )
    return inputs

