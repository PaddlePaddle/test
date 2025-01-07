import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: thresholded_relu
    api简介: thresholded_relu激活函数
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.nn.functional.thresholded_relu(x,  threshold=1.5, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-2 + (3 - -2) * np.random.random([2, 4, 8, 8]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-2 + (3 - -2) * np.random.random([2, 4, 8, 8]).astype('float32'), )
    return inputs

