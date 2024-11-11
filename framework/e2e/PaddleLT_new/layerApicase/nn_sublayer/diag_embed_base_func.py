import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: diag_embed_base
    api简介: 其在指定的 2D 平面（由 dim1 和 dim2 指定）上的对角线由输入 input 填充
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, input, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.nn.functional.diag_embed(input,  offset=0, dim1=-2, dim2=-1, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-2 + (3 - -2) * np.random.random([2, 4, 8, 8, 8]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-2 + (3 - -2) * np.random.random([2, 4, 8, 8, 8]).astype('float32'), )
    return inputs

