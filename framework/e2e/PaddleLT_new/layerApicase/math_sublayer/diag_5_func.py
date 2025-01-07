import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: diag_5
    api简介: 如果 x 是向量（1-D张量），则返回带有 x 元素作为对角线的2-D方阵. 如果 x 是矩阵（2-D张量），则提取 x 的对角线元素，以1-D张量返回
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.diag(x,  offset=1, )
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
    inputs = (paddle.to_tensor(-10 + (10 - -10) * np.random.random([3, 3]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-10 + (10 - -10) * np.random.random([3, 3]).astype('float32'), )
    return inputs

