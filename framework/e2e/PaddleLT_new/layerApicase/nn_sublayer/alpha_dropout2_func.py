import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: alpha_dropout2
    api简介: 一种具有自归一化性质的dropout
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.nn.functional.alpha_dropout(x,  p=1.0, training=False, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(0 + (2 - 0) * np.random.random([2, 3]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (0 + (2 - 0) * np.random.random([2, 3]).astype('float32'), )
    return inputs

