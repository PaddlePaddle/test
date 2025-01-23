import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: AffineTransform_zero_size_func
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.distribution.AffineTransform(
            loc=paddle.to_tensor(-1 + (1 - -1) * np.random.random([12, 3, 0, 10]).astype('float32'), dtype='float32', stop_gradient=False),
            scale=paddle.to_tensor(-1 + (1 - -1) * np.random.random([12, 3, 0, 10]).astype('float32'), dtype='float32', stop_gradient=False),
        ).forward(x=paddle.to_tensor([]))
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

