import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: logical_xor_2
    api简介: 逐元素的对 x 和 y 进行逻辑异或运算
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, y, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.logical_xor(x, y,  )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1), dtype=paddle.bool, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1, -1), dtype=paddle.bool, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(np.random.randint(0, 2, [1, 2]).astype('bool'), dtype='bool', stop_gradient=False), paddle.to_tensor(np.random.randint(0, 2, [2, 2]).astype('bool'), dtype='bool', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.random.randint(0, 2, [1, 2]).astype('bool'), np.random.randint(0, 2, [2, 2]).astype('bool'), )
    return inputs

