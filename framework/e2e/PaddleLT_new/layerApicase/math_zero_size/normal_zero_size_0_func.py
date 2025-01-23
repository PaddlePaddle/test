import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: normal_zero_size_func
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.normal(mean=0.0, std=1.0, shape=[1, 1, 1, 1])
        return out



def create_inputspec(): 
    inputspec = ()
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

