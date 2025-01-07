import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: eye_2
    api简介: 创建形状大小为shape并且数据类型为dtype的Tensor，其中元素值是未初始化的
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.eye( num_rows=paddle.to_tensor([3], dtype='int32', stop_gradient=False), num_columns=paddle.to_tensor([5], dtype='int32', stop_gradient=False), )
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

