import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: normal_base
    api简介: 返回符合正态分布（均值为 mean ，标准差为 std 的正态随机分布）的随机 Tensor
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.normal( shape=[2, 3, 4, 4], )
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

