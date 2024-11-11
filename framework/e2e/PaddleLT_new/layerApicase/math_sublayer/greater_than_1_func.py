import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: greater_than_1
    api简介: 逐元素地返回 x>y 的逻辑值，相同位置前者输入大于后者输入则返回True，否则返回False
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.greater_than(x, y,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor([1], dtype='float32', stop_gradient=False), paddle.to_tensor([1, -1, 2, -4], dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.array([1]).astype('float32'), np.array([1, -1, 2, -4]).astype('float32'), )
    return inputs

