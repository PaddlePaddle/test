import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: one_hot_base
    api简介: 该OP将输入'x'中的每个id转换为一个one-hot向量，其长度为 num_classes
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
    inputs = (paddle.to_tensor(np.random.randint(0, 5, [2, 3, 8, 8]).astype('int32'), dtype='int32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(0, 5, [2, 3, 8, 8]).astype('int32'), )
    return inputs

