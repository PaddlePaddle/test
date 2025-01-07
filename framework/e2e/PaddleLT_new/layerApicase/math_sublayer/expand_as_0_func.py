import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: expand_as_0
    api简介: 根据 y 的形状扩展 x ，扩展后， x 的形状和 y 的形状相同
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.expand_as(x, y,  )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1, -1, -1), dtype=paddle.int32, stop_gradient=True), 
        paddle.static.InputSpec(shape=(-1, -1, -1, -1, -1), dtype=paddle.int32, stop_gradient=True), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(np.random.randint(-1, 1, [2, 3, 1, 3, 1]).astype('int32'), dtype='int32', stop_gradient=False), paddle.to_tensor(np.random.randint(-1, 1, [2, 3, 4, 3, 5]).astype('int32'), dtype='int32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(-1, 1, [2, 3, 1, 3, 1]).astype('int32'), np.random.randint(-1, 1, [2, 3, 4, 3, 5]).astype('int32'), )
    return inputs

