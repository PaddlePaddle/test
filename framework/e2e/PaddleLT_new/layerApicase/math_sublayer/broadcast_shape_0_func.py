import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: broadcast_shape_0
    api简介: 该函数返回对x_shape大小的张量和y_shape大小的张量做broadcast操作后得到的shape
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.broadcast_shape( x_shape=[2, 1, 3, 1, 4, 2, 3], y_shape=[1, 2, 1, 1, 3], )
        out = paddle.to_tensor(out, stop_gradient=False)
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

