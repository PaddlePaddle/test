import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: pad0
    api简介: 该OP依照 pad 和 mode 属性对 x 进行 pad
    """

    def __init__(self):
        super(LayerCase, self).__init__()

    def forward(self, x, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = paddle.nn.functional.pad(x,  pad=paddle.to_tensor(np.random.randint(1, 3, [4]).astype('int32'), dtype='int32', stop_gradient=False), mode='constant', value=0.0, data_format='NCHW', )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([2, 3, 8, 8]).astype('float32'), dtype='float32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([2, 3, 8, 8]).astype('float32'), )
    return inputs

