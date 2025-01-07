import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: full_0
    api简介: 创建形状大小为 shape 并且数据类型为 dtype 的Tensor，其中元素值均为 fill_value
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.full( shape=[2, 3, 4, 4], fill_value=True, dtype='bool', )
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

