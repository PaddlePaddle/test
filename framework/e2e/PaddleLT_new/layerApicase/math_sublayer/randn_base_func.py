import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: randn_base
    api简介: 返回符合标准正态分布（均值为0，标准差为1的正态随机分布）的随机Tensor
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.randn( shape=[2, 3, 4, 4], )
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

