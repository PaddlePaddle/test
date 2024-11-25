import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: BatchNorm3D_4
    api简介: 3维BN批归一化
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.BatchNorm3D(num_features=1, momentum=0.1, epsilon=1e-05, weight_attr=False, bias_attr=False)

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
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 1, 2, 2, 3]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 1, 2, 2, 3]).astype('float32'), )
    return inputs

