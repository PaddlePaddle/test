import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: clip_5
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
        out = paddle.clip(x,  min=paddle.to_tensor([1.0], dtype='float32', stop_gradient=False), )
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
    inputs = (paddle.to_tensor(-1 + (10 - -1) * np.random.random([3, 3]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (10 - -1) * np.random.random([3, 3]).astype('float32'), )
    return inputs

