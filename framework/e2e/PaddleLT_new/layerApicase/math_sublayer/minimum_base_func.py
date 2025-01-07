import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: minimum_base
    api简介: 逐元素对比输入的两个Tensor，并且把各个位置更小的元素保存到返回结果中
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.minimum(x, y,  )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor([1], dtype='float32', stop_gradient=False), paddle.to_tensor([2, -2, 0, 3], dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.array([1]).astype('float32'), np.array([2, -2, 0, 3]).astype('float32'), )
    return inputs

