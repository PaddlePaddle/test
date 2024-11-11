import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: Pad3D_4
    api简介: 3维pad填充
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.Pad3D(padding=[1, 2], mode='constant', data_format='NCL', )

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
        paddle.static.InputSpec(shape=(1, 2, 3), dtype=paddle.float32, stop_gradient=False), 
    )
    return inputspec

def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([1, 2, 3]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([1, 2, 3]).astype('float32'), )
    return inputs

