import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: rand_base
    api简介: 返回符合均匀分布的、范围在[0, 1)的Tensor，形状为 shape，数据类型为 dtype
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.rand( shape=[2, 3, 4, 4], )
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

