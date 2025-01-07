import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: logical_xor_base
    api简介: 逐元素的对 X 和 Y 进行逻辑异或运算
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.logical_xor(x, y,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(np.random.randint(-1, 1, [2, 3, 4, 4]).astype('int32'), dtype='int32', stop_gradient=False), paddle.to_tensor(np.random.randint(-1, 1, [2, 3, 4, 4]).astype('int32'), dtype='int32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(-1, 1, [2, 3, 4, 4]).astype('int32'), np.random.randint(-1, 1, [2, 3, 4, 4]).astype('int32'), )
    return inputs

