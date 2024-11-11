import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: arange_2
    api简介: 该OP返回以步长 step 均匀分隔给定数值区间[start, end)的1-D Tensor
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.arange( start=1, end=20, step=3, )
        return out



def create_inputspec(): 
    inputspec = ( 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = ()
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = ()
    return inputs

