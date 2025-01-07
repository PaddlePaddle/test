import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: SpectralNorm_2
    api简介: 谱归一化
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.SpectralNorm(weight_shape=[2, 4, 5, 6], dim=0, power_iters=1, )

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
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 4, 5, 6]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 4, 5, 6]).astype('float32'), )
    return inputs

