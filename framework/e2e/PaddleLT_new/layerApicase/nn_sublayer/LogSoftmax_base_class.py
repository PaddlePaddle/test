import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: LogSoftmax_base
    api简介: LogSoftmax激活层
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.LogSoftmax(axis=0, )

    def forward(self, data, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = self.func(data, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(2, 2, 1), dtype=paddle.float32, stop_gradient=False), 
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
    inputs = (paddle.to_tensor([[[1], [2]], [[3], [4]]], dtype='float32', stop_gradient=False), )
    return inputs

