import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: expand_1
    api简介: 根据 shape 指定的形状扩展 x ，扩展后， x 的形状和 shape 指定的形状一致
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.expand(x,  shape=paddle.to_tensor([2, 3, 4, 3, 5], dtype='int32', stop_gradient=False), )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 3, 1, 3, 1]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 3, 1, 3, 1]).astype('float32'), )
    return inputs

