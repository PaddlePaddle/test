import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Categorical_zero_size_func
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.distribution.Categorical(logits=paddle.to_tensor([0.1253, 0.5213], dtype='float32')).sample(shape=[10, 0])
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

