import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: SimpleRNN_base
    api简介: 简单循环神经网络
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.SimpleRNN(input_size=3, hidden_size=2, )

    def forward(self, data0, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = self.func(data0, )
        return out



def create_inputspec(): 
    inputspec = ( 
        paddle.static.InputSpec(shape=(-1, -1, -1), dtype=paddle.float32, stop_gradient=False), 
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

