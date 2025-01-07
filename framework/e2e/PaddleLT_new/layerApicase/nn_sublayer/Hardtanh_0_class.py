import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Hardtanh_0
    api简介: Hardtanh激活层
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.Hardtanh(min=-1.5, max=2.0, )

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
        paddle.static.InputSpec(shape=(-1, -1, -1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-3 + (3 - -3) * np.random.random([2, 3, 4, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-3 + (3 - -3) * np.random.random([2, 3, 4, 4, 4]).astype('float32'), )
    return inputs

