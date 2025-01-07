import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: bincount_1
    api简介: 统计输入张量中每个元素出现的次数，如果传入weights张量则每次计数加一时会乘以weights张量对应的值
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, weights, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.bincount(x, weights,  minlength=30, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(np.random.randint(1, 20, [20]).astype('int32'), dtype='int32', stop_gradient=False), paddle.to_tensor(-1 + (1 - -1) * np.random.random([20]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(1, 20, [20]).astype('int32'), -1 + (1 - -1) * np.random.random([20]).astype('float32'), )
    return inputs

