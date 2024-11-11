import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Softmax_4
    api简介: Softmax激活层
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.Softmax(axis=0, )

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
        paddle.static.InputSpec(shape=(-1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
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
    inputs = (paddle.to_tensor([[[2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0], [7.0, 8.0, 8.0, 9.0]], [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [6.0, 7.0, 8.0, 9.0]]], dtype='float32', stop_gradient=False), )
    return inputs

