import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: unique_consecutive_3
    api简介: 将Tensor中连续重复的元素进行去重，返回连续不重复的Tensor
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.unique_consecutive(x,  return_inverse=False, return_counts=False, axis=-2, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-5 + (5 - -5) * np.random.random([3, 4, 5, 5]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-5 + (5 - -5) * np.random.random([3, 4, 5, 5]).astype('float32'), )
    return inputs

