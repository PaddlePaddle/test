import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: diagonal_5
    api简介: 如果输入是 2D Tensor，则返回对角线元素. 如果输入的维度大于 2D，则返回由对角线元素组成的数组
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.diagonal(x,  axis1=-1, axis2=2, )
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
    inputs = (paddle.to_tensor(-5 + (5 - -5) * np.random.random([6, 6, 6, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-5 + (5 - -5) * np.random.random([6, 6, 6, 4, 4]).astype('float32'), )
    return inputs

