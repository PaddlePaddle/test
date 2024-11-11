import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: mm_4
    api简介: 用于两个输入矩阵的相乘, 两个输入的形状可为任意维度，但当任一输入维度大于3时，两个输入的维度必须相等
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, input, mat2, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.mm(input, mat2,  )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1,), dtype=paddle.float32, stop_gradient=False), 
        paddle.static.InputSpec(shape=(-1,), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([7]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-1 + (1 - -1) * np.random.random([7]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([7]).astype('float32'), -1 + (1 - -1) * np.random.random([7]).astype('float32'), )
    return inputs

