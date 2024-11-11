import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: clip_7
    api简介: 向上取整运算函数
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.clip(x,  max=1, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor([[[1, 1, 1], [1, 1, 1]]], dtype='int32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.array([[[1, 1, 1], [1, 1, 1]]]).astype('int32'), )
    return inputs

