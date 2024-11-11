import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: MaxPool2D_8
    api简介: 2维最大池化
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.MaxPool2D(kernel_size=(3, 3), stride=(3, 3), padding=(0, 0), )

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
        paddle.static.InputSpec(shape=(-1, -1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 3, 32, 32]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 3, 32, 32]).astype('float32'), )
    return inputs

