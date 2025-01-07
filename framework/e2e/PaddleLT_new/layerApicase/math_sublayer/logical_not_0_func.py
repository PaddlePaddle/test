import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: logical_not_0
    api简介: 逐元素的对 X Tensor进行逻辑非运算
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.logical_not(x,  )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1, -1), dtype=paddle.bool, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(np.random.randint(0, 2, [2, 3, 4, 4]).astype('bool'), dtype='bool', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(0, 2, [2, 3, 4, 4]).astype('bool'), )
    return inputs

