import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Conv3DTranspose_0
    api简介: 3维反卷积
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.Conv3DTranspose(kernel_size=[2, 2, 2], in_channels=3, out_channels=1, stride=2, padding=1, output_padding=1, groups=1, dilation=1, )

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
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 3, 4, 4, 4]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 3, 4, 4, 4]).astype('float32'), )
    return inputs

