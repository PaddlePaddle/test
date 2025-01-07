import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Conv3D_14
    api简介: 3维卷积
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.Conv3D(in_channels=3, out_channels=1, kernel_size=[3, 3, 3], stride=[2, 2, 1], padding=(1, 2, 2), dilation=(2, 2, 2), padding_mode='zeros', )

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
        paddle.static.InputSpec(shape=(-1, 3, -1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(0 + (1 - 0) * np.random.random([2, 3, 4, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (0 + (1 - 0) * np.random.random([2, 3, 4, 4, 4]).astype('float32'), )
    return inputs

