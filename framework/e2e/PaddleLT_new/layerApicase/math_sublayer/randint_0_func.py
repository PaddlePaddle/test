import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: randint_0
    api简介: 返回服从均匀分布的、范围在[low, high)的随机Tensor
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.randint( low=2, high=5, shape=[2, 3, 4, 4], )
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

