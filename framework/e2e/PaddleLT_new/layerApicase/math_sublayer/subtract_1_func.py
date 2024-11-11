import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: subtract_1
    api简介: 逐元素相减算子，输入 x 与输入 y 逐元素相减，并将各个位置的输出元素保存到返回结果中
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.subtract(x, y,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor([2, '-inf', 5], dtype='float32', stop_gradient=False), paddle.to_tensor([1, 4, 'inf'], dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.array([2, '-inf', 5]).astype('float32'), np.array([1, 4, 'inf']).astype('float32'), )
    return inputs

