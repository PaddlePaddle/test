import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: atan2_1
    api简介: 对x/y进行逐元素的arctangent运算，通过符号确定象限
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.atan2(x, y,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(np.random.randint(-5, 5, [6, 6]).astype('int32'), dtype='int32', stop_gradient=False), paddle.to_tensor(np.random.randint(-5, 5, [6, 6]).astype('int32'), dtype='int32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(-5, 5, [6, 6]).astype('int32'), np.random.randint(-5, 5, [6, 6]).astype('int32'), )
    return inputs

