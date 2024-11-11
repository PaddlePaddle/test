import numpy as np
import paddle


class LayerCase(paddle.nn.Layer):
    """
    case名称: SimpleRNN_5
    api简介: 简单循环神经网络
    """

    def __init__(self):
        super(LayerCase, self).__init__()
        self.func = paddle.nn.SimpleRNN(input_size=4, hidden_size=4, dropout=0.8, )

    def forward(self, data0, data1, data2, ):
        """
        forward
        """

        paddle.seed(33)
        np.random.seed(33)
        out = self.func(data0, data1, data2, )
        return out


def create_tensor_inputs():
    """
    paddle tensor
    """
    inputs = (paddle.to_tensor(-1 + (1 - -1) * np.random.random([1, 2, 4]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(-1 + (1 - -1) * np.random.random([1, 1, 4]).astype('float32'), dtype='float32', stop_gradient=False), paddle.to_tensor(np.random.randint(2, 4, [1]).astype('int32'), dtype='int32', stop_gradient=False), )
    return inputs


def create_numpy_inputs():
    """
    numpy array
    """
    inputs = (-1 + (1 - -1) * np.random.random([1, 2, 4]).astype('float32'), -1 + (1 - -1) * np.random.random([1, 1, 4]).astype('float32'), np.random.randint(2, 4, [1]).astype('int32'), )
    return inputs

