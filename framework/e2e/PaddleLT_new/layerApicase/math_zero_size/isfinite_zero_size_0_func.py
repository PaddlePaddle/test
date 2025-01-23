import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: isfinite_zero_size_func
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.isfinite(x,  )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor([[-1.0, 2.0, 'nan'], ['-inf', 'inf', '-nan'], [2.4, 0.0, '-inf']], dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (np.array([[-1.0, 2.0, 'nan'], ['-inf', 'inf', '-nan'], [2.4, 0.0, '-inf']]).astype('float32'), )
    return inputs

